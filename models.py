import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation, TransformerEncoder


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


### Generator module creates fake annotation (node) and adjancency (edge) matrices. 

### Generator takes a randomly generated tensor from prior (see solver.py-sample_z function) and 
### passes it through transformer encoder. 

### At the end of generator annotation (nodes_logits) and adjancency (edges_logits) are produced. 

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, feature_matching, conv_dims, z_dim, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio, drop_rate):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate
        self.feature_matching = feature_matching
        ### After parsing the arguments generaot network will be created. 
        
        ### layers list holds the sequential layers which processes tensor that is coming from prior. 
        ### Dimension of the tensor changes as follows:
        ### (batch_size, z_dim) -> (batch_size, 128) -> (batch_size, 256) -> (batch_size, 512)

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)
        
        ### After passing through MLP, node and edge tensors are created seperately.
        ### Dimension for nodes and edge tensors are as follows:
        
        ### node: (batch_size, 512) -> (batch_size, vertexes*dim)
        ### edge: (batch_size, 512) -> (batch_size, vertexes*vertexes*dim)
        self.edges_layer = nn.Linear(conv_dims[-1], vertexes * vertexes * dim)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * dim)
        self.features_layer = nn.Linear(conv_dims[-1], vertexes * dim)
        self.dropoout = nn.Dropout(p=dropout)
        


        
        ### After handling the dimensions, tensors are passed through Transformer encoder block. 
        ### Here you can see that data are split to 4 GPUs. So now, 
        ### Transformers handle tensors with dimensions of (batch_size/4, node.shape[1]) or (batch_size/4, edge.shape[1])
        
        
        self.Transformer = torch.nn.DataParallel(TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
                                                                    mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate),device_ids=[0,1,2,3])
        
        ### After Transformer encoder, final dimension adjusment are done with linear layer. 
        
        ### node: (batch_size, vertexes*dim) -> (batch_size, vertexes, nodes)
        ### edge: (batch_size, vertexes*vertexes*dim) -> (batch_size, vertexes, vertexes, edges)
        
        self.nodes_output_layer = nn.Linear(self.dim, self.nodes)
        self.edges_output_layer = nn.Linear(self.dim, self.edges)
        
        
    def forward(self, x):
      
        output = self.layers(x)

        #### EDGE LOGITS ####
        
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.dim,self.vertexes,self.vertexes)
               
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
      
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1)) ### (16,9*9,512)
        edges_logits = edges_logits.view(-1, self.vertexes*self.vertexes,self.dim) ### (16, 81, 512)
        edges_logits = self.Transformer(edges_logits)
        edges_logits = self.dropoout(edges_logits)
        edges_logits = edges_logits.view(-1,self.vertexes,self.vertexes,self.dim)  ### (16,9,9,512)
        edges_logits = self.edges_output_layer(edges_logits) ### (16,9,9,5)
        
        
        #### NODE LOGITS ####
        
        nodes_logits = self.nodes_layer(output)  ### (16,9*512)
     
        nodes_logits = nodes_logits.view(-1,self.vertexes,self.dim) ### (16,9,512)
        nodes_logits = self.Transformer(nodes_logits) 
        nodes_logits = self.dropoout(nodes_logits)
        nodes_logits = nodes_logits.view(-1,self.vertexes,self.dim)  ### (16,9,512)
        nodes_logits = self.nodes_output_layer(nodes_logits) ## (16,9,5)
        
        
       
               
        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, m_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        
        adj = adj[:,:,:,1:].permute(0,3,1,2)

        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)
        
        # Need to implement batch discriminator #
        #########################################

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output
        
        return output, h
    
