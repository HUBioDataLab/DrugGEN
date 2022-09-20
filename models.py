import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import PNA, GraphConvolution, GraphAggregation, TransformerEncoder, TransformerDecoder
import copy 
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.dense import DenseGCNConv

    
    
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio, drop_rate):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate
        dropout= self.dropout_rate
    

        
        self.layers_edge = nn.Sequential(nn.Linear(self.edges, conv_dims[0]), nn.ReLU(), nn.Linear(conv_dims[0], self.dim))  #  128, 9, 9, 5 --> 128, 9, 9, 128
        
        self.layers_node = nn.Sequential(nn.Linear(self.nodes, conv_dims[0]), nn.ReLU(), nn.Linear(conv_dims[0], self.dim))  #  128, 9, 5 --> 128, 9, 128
        
    
        
        self.TransformerEncoder = TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
                                                                    mlp_ratio=self.mlp_ratio, drop_rate=self.dropout_rate)         
     

        self.dropout = nn.Dropout(p=dropout)
        
        self.nodes_output_layer = nn.Linear(self.dim, self.nodes)  # 128 -- > 5
        self.edges_output_layer = nn.Linear(self.dim, self.edges)  # 128 -- > 5
      

    def forward(self, z_e,z_n,a,a_tensor,x_tensor):
        
        nodes_logits = self.layers_node(z_n) 
        
        nodes_logits = self.dropout(nodes_logits)
        
        edges_logits = self.layers_edge(z_e)
        
        edges_logits = edges_logits.view(-1,self.dim,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropout(edges_logits)
        edges_logits = edges_logits.view(-1,self.vertexes,self.vertexes,self.dim)


        nodes_logits , edges_logits, attn = self.TransformerEncoder(nodes_logits, edges_logits)
        
        edges_logits = self.dropout(edges_logits)
        nodes_logits = self.dropout(nodes_logits)
        
        
        nodes_logits = self.nodes_output_layer(nodes_logits)
        edges_logits = self.edges_output_layer(edges_logits)

        return edges_logits, nodes_logits, attn
        
class Generator2(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, drop_rate,drugs_m_dim,drugs_b_dim,b_dim,m_dim):
        super().__init__()

        self.depth = depth
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.heads = heads
        self.dropout_rate = drop_rate
        self.drugs_m_dim = drugs_m_dim
        self.drugs_b_dim = drugs_b_dim
        self.prot_n = 7
        self.prot_e = 1 
    
        self.dropoout = nn.Dropout(p=drop_rate)
        
        self.mol_nodes = nn.Linear(m_dim, dim)
        self.mol_edges = nn.Linear(b_dim, dim)
        
        self.prot_nodes =  nn.Linear(self.prot_n, dim)
        self.prot_edges =  nn.Linear(self.prot_e, dim)
        
        self.TransformerDecoder = TransformerDecoder(dim, depth, heads, mlp_ratio=4, drop_rate=0.)

        self.nodes_output_layer = nn.Linear(self.dim, self.drugs_m_dim)
        self.edges_output_layer = nn.Linear(self.dim, self.drugs_b_dim)


    def forward(self, edges_logits, nodes_logits, prot_n, prot_e):
        
        edges_logits = self.mol_edges(edges_logits)
        nodes_logits = self.mol_nodes(nodes_logits)
        
        prot_n = self.prot_nodes(prot_n)
        prot_e = self.prot_edges(prot_e)
        
        edges_logits, nodes_logits, dec_attn = self.TransformerDecoder(nodes_logits,prot_n,edges_logits,prot_e)
     
        edges_logits = self.edges_output_layer(edges_logits)
        nodes_logits = self.nodes_output_layer(nodes_logits)
        
        edges_logits = self.dropoout(edges_logits)
        nodes_logits = self.dropoout(nodes_logits)
        
        return edges_logits, nodes_logits, dec_attn

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,deg,agg,sca,pna_in_ch,pna_out_ch,edge_dim,towers,pre_lay,post_lay,pna_layer_num, graph_add):
        super(Discriminator, self).__init__()
        self.degree = deg
        self.aggregators = agg
        self.scalers = sca
        self.pna_in_channels = pna_in_ch
        self.pna_out_channels = pna_out_ch
        self.edge_dimension = edge_dim
        self.towers = towers
        self.pre_layers_num = pre_lay
        self.post_layers_num = post_lay
        self.pna_layer_num = pna_layer_num
        self.graph_add = graph_add
        self.PNA_layer = PNA(deg=self.degree, agg =self.aggregators,sca = self.scalers,
                             pna_in_ch= self.pna_in_channels, pna_out_ch = self.pna_out_channels, edge_dim = self.edge_dimension,
                             towers = self.towers, pre_lay = self.pre_layers_num, post_lay = self.post_layers_num,
                             pna_layer_num = self.pna_layer_num, graph_add = self.graph_add)

    def forward(self, x, edge_index, edge_attr, batch, activation=None):

        h = self.PNA_layer(x, edge_index, edge_attr, batch)

        h = activation(h) if activation is not None else h
        
        return h

class Discriminator2(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,deg,agg,sca,pna_in_ch,pna_out_ch,edge_dim,towers,pre_lay,post_lay,pna_layer_num, graph_add):
        super(Discriminator2, self).__init__()
        self.degree = deg
        self.aggregators = agg
        self.scalers = sca
        self.pna_in_channels = pna_in_ch
        self.pna_out_channels = pna_out_ch
        self.edge_dimension = edge_dim
        self.towers = towers
        self.pre_layers_num = pre_lay
        self.post_layers_num = post_lay
        self.pna_layer_num = pna_layer_num
        self.graph_add = graph_add
        self.PNA_layer = PNA(deg=self.degree, agg =self.aggregators,sca = self.scalers,
                             pna_in_ch= self.pna_in_channels, pna_out_ch = self.pna_out_channels, edge_dim = self.edge_dimension,
                             towers = self.towers, pre_lay = self.pre_layers_num, post_lay = self.post_layers_num,
                             pna_layer_num = self.pna_layer_num, graph_add = self.graph_add)

    def forward(self, x, edge_index, edge_attr, batch, activation=None):

        h = self.PNA_layer(x, edge_index, edge_attr, batch)

        h = activation(h) if activation is not None else h
        
        return h


class Discriminator_old(nn.Module):

    def __init__(self, conv_dim, m_dim, b_dim, dropout, gcn_depth):
        super(Discriminator_old, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout,gcn_depth)
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
    
class Discriminator_old2(nn.Module):

    def __init__(self, conv_dim, m_dim, b_dim, dropout, gcn_depth):
        super(Discriminator_old2, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout, gcn_depth)
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
    
class Discriminator3(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,in_ch):
        super(Discriminator3, self).__init__()
        self.dim = in_ch
        
        
        self.TraConv_layer = TransformerConv(in_channels = self.dim,out_channels =  self.dim//4,edge_dim = self.dim)  
        self.mlp = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(self.dim//4,1))
    def forward(self, x, edge_index, edge_attr, batch, activation=None):

        h = self.TraConv_layer(x, edge_index, edge_attr)
        h = global_add_pool(h,batch)
        h = self.mlp(h)
        h = activation(h) if activation is not None else h
        
        return h
    
    
class PNA_Net(nn.Module):
    def __init__(self,deg):
        super().__init__()

      

        self.convs = nn.ModuleList()
        
        self.lin = nn.Linear(5, 128)
        for _ in range(1):
            conv = DenseGCNConv(128, 128, improved=False, bias=True)
            self.convs.append(conv)
            
        self.agg_layer = GraphAggregation(128, 128, 0, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(),
                              nn.Linear(32, 1))

    def forward(self, x, adj,mask=None):
        x = self.lin(x)
        
        for conv in self.convs:
            x = F.relu(conv(x, adj,mask=None))

        x = self.agg_layer(x,torch.tanh)
       
        return self.mlp(x)     