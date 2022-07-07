import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import PNA, GraphConvolution, GraphAggregation
import copy 
from transformer import GraphTransformerEncoder, Encoder, EncoderLayer, Node_Embeddings, Edge_Embeddings, MultiHeadedAttention, PositionwiseFeedForward, GraphTransformerDecoder, MoleculeEncoderDecoderLayer, Decoder, MultiHeadedAttentionDecoder, PositionwiseFeedForwardDecoder, TransformerDecoderAttention, Node_Embeddings_dec, Edge_Embeddings_dec, Position_Encoding_dec
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import TransformerConv

    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio, drop_rate, tra_conv):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate
        N=self.depth
        d_model=self.dim
        h=self.heads
        dropout= self.dropout_rate
        attenuation_lambda=0.1
        max_length=self.vertexes
        N_dense=2
        leaky_relu_slope=0.0 
        dense_output_nonlinearity='tanh' 
        scale_norm=False
        d_atom = self.nodes
        d_edge = self.edges
        c = copy.deepcopy
        self.tra_conv = tra_conv
        output_layer = vertexes * vertexes * dim + vertexes * dim
        
        self.layers_edge = nn.Sequential(nn.Linear(self.vertexes, conv_dims[0]), nn.Tanh(), nn.Linear(conv_dims[0], self.vertexes*self.edges),
                                         nn.Sigmoid(), nn.Dropout(p=dropout))
        
        self.layers_node = nn.Sequential(nn.Linear(self.vertexes, conv_dims[0]), nn.Tanh(), nn.Linear(conv_dims[0], self.vertexes*self.nodes),
                                         nn.Sigmoid(), nn.Dropout(p=dropout))

        #self.layers = nn.Sequential(nn.Linear(vertexes, conv_dims[0]), nn.Tanh(), nn.Linear(conv_dims[0], conv_dims[1]), nn.Tanh(), 
         #                                nn.Tanh(), nn.Linear(conv_dims[1], vertexes * (vertexes+1)),
          #                               nn.Sigmoid(), nn.Dropout(p=dropout))
        
        attn = MultiHeadedAttention(h, d_model, leaky_relu_slope, dropout, attenuation_lambda)
        ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity) 
        
        #self.Transformer_n = torch.nn.DataParallel(TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
         #                                                           mlp_ratio=self.mlp_ratio, drop_rate=self.dropout_rate), device_ids=[0,1])              
        #self.Transformer_e = torch.nn.DataParallel(TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads,
         #                                                           mlp_ratio=self.mlp_ratio, drop_rate=self.dropout_rate), device_ids=[0,1])        
        
        self.TransformerEncoder = GraphTransformerEncoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
                                            Node_Embeddings(d_atom, d_model, dropout),
                                            Edge_Embeddings(d_edge, d_model, dropout),dim)


        self.dropout = nn.Dropout(p=dropout)
        self.last_dropout= nn.Dropout(p=0.99)
        self.nodes_output_layer = nn.Sequential(nn.Linear(self.dim, self.nodes))
        self.edges_output_layer = nn.Sequential(nn.Linear(self.dim, self.edges))
        if self.tra_conv: 
            self.attr_mlp = nn.Linear(1, self.dim)
            self.nodes_mlp = nn.Linear(1, self.dim)
        
    def postprocess(self, inputs, post_method, temperature=1.,dimension=-1):
        
        if post_method == 'soft_gumbel':
            softmax = F.gumbel_softmax(inputs
                        / temperature, hard=False, dim = dimension)
        elif post_method == 'hard_gumbel':
            softmax = F.gumbel_softmax(inputs
                        / temperature, hard=True, dim = dimension)
        elif post_method == 'softmax':
            softmax = F.softmax(inputs / temperature, dim = dimension)
            
        return softmax
    
    def dense_to_sparse_with_attr(self, adj):
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]

        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])
            index = torch.stack(index, dim=0)
        return index, edge_attr.long()
      
        
    def forward(self, z_e,z_n,a,a_tensor,x_tensor):
        
        nodes_logits = self.layers_node(z_n) 
        nodes_logits = nodes_logits.view(-1,self.vertexes,self.nodes)

        edges_logits = self.layers_edge(z_e)
        edges_logits = edges_logits.view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropout(edges_logits)
        

   
 
        adj_matrix = torch.max(edges_logits,1)[1]
        edges_logits = edges_logits.view(-1,self.vertexes,self.vertexes,self.edges)
        
        nodes_logits , edges_logits = self.TransformerEncoder(nodes_logits, adj_matrix, edges_logits)
        
        edges_logits = self.dropout(edges_logits)
        nodes_logits = self.dropout(nodes_logits)
        
        
        nodes_logits_sample = self.nodes_output_layer(nodes_logits)
        # 128,25,13
        
        edges_logits_sample = self.edges_output_layer(edges_logits)  
        # 128,25,25,5
        
        edges_logits_sample = edges_logits_sample.view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits_sample = self.postprocess(edges_logits_sample, "soft_gumbel", dimension=1)
        nodes_logits_sample = self.postprocess(nodes_logits_sample, "soft_gumbel")
        
        
        
        edges_hat, nodes_hat = torch.max(edges_logits_sample, 1)[1], torch.max(nodes_logits_sample, -1)[1] 
        
        
        fake_edge_index, fake_edge_attr = self.dense_to_sparse_with_attr(edges_hat)
        nodes_fake = nodes_hat.view(-1,1)
        if self.tra_conv:
            attr_for_traconv = self.attr_mlp(fake_edge_attr.view(-1,1).float())
            nodes_for_traconv = self.nodes_mlp(nodes_fake.view(-1,1).float())
        if self.tra_conv:
            return edges_logits_sample, nodes_logits_sample, edges_logits, nodes_logits,nodes_fake,fake_edge_index,fake_edge_attr, attr_for_traconv, nodes_for_traconv
        else:
            return edges_logits_sample, nodes_logits_sample, edges_logits, nodes_logits,nodes_fake,fake_edge_index,fake_edge_attr
        
class Generator2(nn.Module):
    def __init__(self, vertexes_mol, edges_mol, nodes_mol, vertexes_protein, 
                edges_protein, nodes_protein, dropout, dim, depth, heads, mlp_ratio, 
                drop_rate,drugs_m_dim,drugs_b_dim):
        super().__init__()

        ## edge_logits = 16,9,9,4 (bs, mol_length, mol_length, bond_type)

        #self.conv_dims = conv_dims
        self.vertexes_prot = vertexes_protein # protein_length = 
        self.edges_prot = 1 # bond_type_number =
        self.nodes_prot = 7 # atom_type_num =
        self.vertexes_mol = vertexes_mol
        self.edges_mol = edges_mol 
        self.nodes_mol = nodes_mol
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate
        N=self.depth
        d_model=self.dim
        h=self.heads
        dropout= self.dropout_rate
        attenuation_lambda=0.1
        max_length= 546
        N_dense=2
        leaky_relu_slope=0.0 
        dense_output_nonlinearity='tanh' 
        scale_norm=False
        d_atom = self.nodes_prot
        d_edge = self.edges_prot
        c = copy.deepcopy
        self.drugs_m_dim = drugs_m_dim
        self.drugs_b_dim = drugs_b_dim
        #self.prot_e_layer = nn.Sequential(nn.Linear(self.edges_prot, conv_dims[0]), nn.Tanh(), nn.Linear(conv_dims[0], conv_dims[1]), nn.Tanh(), nn.Linear(conv_dims[1], conv_dims[2]),
         #                                nn.Tanh(), nn.Linear(conv_dims[2], self.dim),
          #                               nn.Tanh(), nn.Dropout(p=dropout))
        #self.prot_n_layer = nn.Sequential(nn.Linear(self.nodes_prot, conv_dims[0]), nn.Tanh(), nn.Linear(conv_dims[0], conv_dims[1]), nn.Tanh(), nn.Linear(conv_dims[1], conv_dims[2]),
         #                                nn.Tanh(), nn.Linear(conv_dims[2], self.dim),
          #                               nn.Tanh(), nn.Dropout(p=dropout))        
        self.dropoout = nn.Dropout(p=dropout)

        attn = MultiHeadedAttentionDecoder(h, d_model, leaky_relu_slope, dropout, attenuation_lambda)
        ff = PositionwiseFeedForwardDecoder(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity) 
        dec_attn = TransformerDecoderAttention(heads, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1)
        self.TransformerDecoder = GraphTransformerDecoder(Decoder(MoleculeEncoderDecoderLayer(d_model, c(attn), c(ff), dropout, scale_norm, c(dec_attn)), N, scale_norm),
                                            Node_Embeddings(d_atom, d_model, dropout),
                                            Edge_Embeddings(d_edge, d_model, dropout))


        self.nodes_output_layer = nn.Linear(self.dim, self.drugs_m_dim)
        self.edges_output_layer = nn.Linear(self.dim, self.drugs_b_dim)
        self.attr_mlp = nn.Linear(1, self.dim)
        self.nodes_mlp = nn.Linear(1, self.dim)
    def postprocess(self, inputs, post_method, temperature=1.,dimension=-1):
        
        if post_method == 'soft_gumbel':
            softmax = F.gumbel_softmax(inputs
                        / temperature, hard=False, dim = dimension)
        elif post_method == 'hard_gumbel':
            softmax = F.gumbel_softmax(inputs
                        / temperature, hard=True, dim = dimension)
        elif post_method == 'softmax':
            softmax = F.softmax(inputs / temperature, dim = dimension)
            
        return softmax
    
    
    def dense_to_sparse_with_attr(self, adj):
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]

        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])
            index = torch.stack(index, dim=0)
        return index, edge_attr.long()
    
    def forward(self, edges_logits, nodes_logits, prot_n, prot_e):

        ##### EDGE LOGITS #####
        adj_matrix = torch.max(edges_logits,-1)[1]
        
        
        edges_logits, nodes_logits, attn = self.TransformerDecoder(edges_logits, nodes_logits, prot_n, prot_e,adj_matrix)
   
        
        ##### NODE LOGITS #####
        
        edges_hat = self.edges_output_layer(edges_logits)
        nodes_hat = self.nodes_output_layer(nodes_logits)
        edges_hat = edges_hat.view(-1,self.edges_mol, self.vertexes_mol,self.vertexes_mol)
        edges_hat = self.postprocess(edges_hat, "soft_gumbel", dimension=1)
        nodes_hat = self.postprocess(nodes_hat, "soft_gumbel")        
        
        edges_hard, nodes_hard = torch.max(edges_hat, 1)[1], torch.max(nodes_hat, -1)[1] 
        
        edges_hat = edges_hat.view(-1, self.vertexes_mol,self.vertexes_mol,self.edges_mol)
        fake_edge_index2, fake_edge_attr2 = self.dense_to_sparse_with_attr(edges_hard)
        
        nodes_fake2 = nodes_hard.view(-1,1)
        g_attr_for_traconv = self.attr_mlp(fake_edge_attr2.view(-1,1).float())
        g_nodes_for_traconv = self.nodes_mlp(nodes_fake2.view(-1,1).float())        
        return edges_hat, nodes_hat, edges_hard, nodes_hard, nodes_fake2, fake_edge_index2, fake_edge_attr2,g_attr_for_traconv,g_nodes_for_traconv

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

    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator_old, self).__init__()

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
    
class Discriminator_old2(nn.Module):

    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator_old2, self).__init__()

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