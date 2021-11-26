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



class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio, drop_rate):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)
        
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)
        
        #self.Transformer = TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
                                                                    #mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        
        self.Transformer = torch.nn.DataParallel(TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
                                                                    mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate))
        #self.Transformer = torch.nn.DataParallel(self.Transformer)


    def forward(self, x):
        #print(self.layers)
        #print("noise:", type(x))
        #print("noise:", x.shape)
        output = self.layers(x)
        #print("x:", type(output))
        #print("x:", output.shape)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes,self.dim)
        #print("edge shape:" ,edges_logits.shape)     
               
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2,4))/2
        #print("edge shape:" ,edges_logits.shape)
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1,4))
        #print("edge shape:" ,edges_logits.shape)
        nodes_logits = self.nodes_layer(output)
        #print("x before:", type(nodes_logits))
        #print("x before:", nodes_logits.shape)
        nodes_logits = nodes_logits.view(-1,self.vertexes*self.nodes,self.dim) # (16,9,5,720)
        #print("x after:", nodes_logits.shape)
        #nodes_logits = nodes_logits.view(-1, self.node_dim, self.dim) #16,9,5
        #print("x after:", type(nodes_logits))
        #print("x after:", nodes.shape)
        nodes_logits = self.Transformer(nodes_logits) #(16,45,720)
        nodes_logits = self.dropoout(nodes_logits)
        nodes_logits = nodes_logits.view(-1,self.vertexes,self.nodes,self.dim)  # (16,9,5,720)
        #print("x after_trans:", type(nodes_logits))
        #print("x after_trans:", nodes_logits.shape)
       #print("edge shape:" ,edges_logits.shape)
        edges_logits = edges_logits.contiguous().view(-1, self.vertexes*self.vertexes*self.edges,self.dim)
        edges_logits = self.Transformer(edges_logits)
        edges_logits = self.dropoout(edges_logits)
        edges_logits = edges_logits.view(-1,self.vertexes,self.vertexes,self.edges,self.dim)
        #print("edges after trans:", edges_logits.shape)
        #print("nodes_logits:",type(nodes_logits))
        #nodes_logits = nodes_logits.view(self.dim)
        #print("nodes_logits:", nodes_logits.shape)
        #nodes_logits = self.Transformer_node(nodes_logits)

        #nodes_logits = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))
        #edges_logits = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))


        nodes_logits = torch.mean(nodes_logits,-1) #(16,9,5,720)  -------> (16,9,5)
        edges_logits = torch.mean(edges_logits,-1) #(16,9,9,5,720) ------> (16,9,9,5)
        #print("last_nodes: ",nodes_logits.shape)
        #print("last_edges: ", edges_logits.shape)


        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

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
        
        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output
        
        return output, h