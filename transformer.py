import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerEncoder(nn.Module):
    def __init__(self, encoder, node_embed, edge_embed,dim):
        super(GraphTransformerEncoder, self).__init__()
        self.encoder = encoder
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        #self.pos_embed = pos_embed
        self.pos = nn.Linear(3,dim)
        self.dropout = nn.Dropout(0.1)
        
    def laplacian_positional_encoding(self, x):
        
        A = x
     
        N = torch.count_nonzero(A, -1)

        L = N - A
 

        EigVal, EigVec = np.linalg.eig(L)
        
        idx = np.argsort(EigVal) # increasing order
    
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        
        pos_enc = EigVec[:,0:3]
      
        return torch.from_numpy(pos_enc).to("cuda").float()
    
    def forward(self, node_features,  adj_matrix, edge_features):
        """Take in and process src and target sequences."""
        # return self.predict(self.encode(src,  adj_matrix, edges_att))
        return self.encode(node_features, edge_features, adj_matrix)

    def encode(self, node_features, edge_features, adj_matrix):  # (batch, max_length, d_atom+1)
        
        # xv.shape = (batch, max_length, d_model)
        adj_matrix_cpu = adj_matrix.to("cpu")
     
        pos_enc = [self.laplacian_positional_encoding(adj_matrix_cpu[i]) for i in range(adj_matrix_cpu.shape[0])]

        pos_enc = torch.stack(pos_enc)


        pos_embeding = self.pos(self.dropout(pos_enc))
        node_embeding = self.node_embed(node_features)

   
        node_initial = pos_embeding + node_embeding
        # node_initial = self.node_embed(node_features[:, :, :-1])
        # evw = xv + evw for directions; evw.shape = (batch, max_length, max_length, d_model)
        # edge_initial = node_initial.unsqueeze(-2) + self.edge_embed(edge_features)
        edge_initial = self.edge_embed(edge_features)
        return self.encoder(node_initial, edge_initial, adj_matrix)



# Embeddings


class Node_Embeddings(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb
        self.d_atom = d_atom
    def forward(self, x):  # x.shape(batch, max_length, d_atom)


        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)


class Edge_Embeddings(nn.Module):
    def __init__(self, d_edge, d_emb, dropout):
        super(Edge_Embeddings, self).__init__()
        self.lut = nn.Linear(d_edge, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape = (batch, max_length, max_length, d_edge)
    
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)

"""
class Position_Encoding(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding, self).__init__()
        self.max_length = max_length 
        self.d_emb = d_emb
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Linear(max_length, d_emb)  
    
    
    def forward(self, x):
        
        
        return self.pe(self.dropout(x))  # (batch, max_length) -> (batch, max_length, d_emb)
"""

# Generator
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def swish_function(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish_function(x):
    return x * torch.tanh(F.softplus(x))


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, node_hidden, edge_hidden, adj_matrix):
        """Pass the input through each layer in turn."""
        for layer in self.layers:
            node_hidden, edge_hidden = layer(node_hidden, edge_hidden, adj_matrix)
            node_hidden, edge_hidden = self.norm(node_hidden), self.norm(edge_hidden)
            

        return node_hidden, edge_hidden


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn        # MultiHeadedAttention
        self.feed_forward = feed_forward  # PositionwiseFeedForward
        # self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, node_hidden, edge_hidden, adj_matrix):
        
        """Follow Figure 1 (left) for connections."""
        
        # x.shape = (batch, max_length, d_atom)
        
        node_hidden = self.dropout(self.norm(node_hidden))
        edge_hidden = self.dropout(self.norm(edge_hidden))
        
        node_hidden_first, edge_hidden_first = self.self_attn(node_hidden, node_hidden, edge_hidden, adj_matrix)
        
        # the first residue block
        
        edge_hidden_first = edge_hidden + self.dropout(self.norm(edge_hidden_first))
        
        edge_hidden_second = self.feed_forward(edge_hidden_first)
        
        node_hidden_first = node_hidden + self.dropout(self.norm(node_hidden_first))
        
        node_hidden_second = self.feed_forward(node_hidden_first)
        
        # the second residue block
        
        node_hidden_third = node_hidden_first + self.dropout(self.norm(node_hidden_second))
        
        edge_hidden_third = edge_hidden_first + self.dropout(self.norm(edge_hidden_second))
        
        return node_hidden_third, edge_hidden_third

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.1, dense_output_nonlinearity='tanh'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'gelu':
            self.dense_output_nonlinearity = lambda x: F.gelu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
        elif dense_output_nonlinearity == 'swish':
            self.dense_output_nonlinearity = lambda x: x * torch.sigmoid(x)
        elif dense_output_nonlinearity == 'mish':
            self.dense_output_nonlinearity = lambda x: x * torch.tanh(F.softplus(x))

    def forward(self, node_hidden):
        if self.N_dense == 0:
            return node_hidden

        for i in range(self.N_dense - 1):
            node_hidden = self.dropout[i](mish_function(self.linears[i](node_hidden)))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](node_hidden)))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


# Attention


def attention(query, key, value, adj_matrix, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # query.shape = (batch, h, max_length, d_e)
    # key.shape = (batch, h, max_length, max_length, d_e)
    # value.shape = (batch, h, max_length, d_e)
    # out_scores.shape = (batch, h, max_length, max_length)
    # in_scores.shape = (batch, h, max_length, max_length)
    # normal einsum out = 'bhmd,bhmnd->bhmn' , normal einsum in = 'bhnd,bhmnd->bhnm', normal node = 'bhmn,bhnd->bhmd'
    d_e = query.size(-1)
    out_scores = torch.einsum('bhmd,bhmnd->bhmn', query, key) / math.sqrt(d_e)
    in_scores = torch.einsum('bhnd,bhmnd->bhnm', query, key) / math.sqrt(d_e)

    
    
    out_attn = F.softmax(out_scores, dim=-1)
    in_attn = F.softmax(in_scores, dim=-1)
    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message = out_attn + in_attn - diag_attn
    

    # add the diffusion caused by distance
    #message = message * adj_matrix.unsqueeze(1)

    if dropout is not None:
        message = dropout(message)

    # message.shape = (batch, h, max_length, max_length), value.shape = (batch, h, max_length, d_k)
    node_hidden = torch.einsum('bhmn,bhnd->bhmd', message, value)
    
    
    
    edge_hidden = message.unsqueeze(-1) * key
    
    
    return node_hidden, edge_hidden, message


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads  # We assume d_v always equals d_k
        self.heads = heads

        #self.atten_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda), requires_grad=True)

        self.linears = clones(nn.Linear(d_model, d_model), 5)  # 5 for query, key, value, node update, edge update

        self.message = None
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

        
        self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
    

    def forward(self, query_node, value_node, key_edge, adj_matrix):
        """Implements Figure 2"""
        
        n_batches, max_length, d_model = query_node.shape

        # 1) Prepare adjacency matrix with shape (batch, max_length, max_length)
        #torch.clamp(self.atten_lambda, min=0, max=1)
        #adj_matrix = self.atten_lambda * adj_matrix
        
        adj_matrix = self.distance_matrix_kernel(adj_matrix.float())

        # 2) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query_node).view(n_batches, max_length, self.heads, self.d_k).transpose(1, 2)
        key = self.linears[1](key_edge).view(n_batches, max_length, max_length, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        value = self.linears[2](value_node).view(n_batches, max_length, self.heads, self.d_k).transpose(1, 2)
        
        # 3) Apply attention on all the projected vectors in batch.
        node_hidden, edge_hidden, self.message = attention(query, key, value, adj_matrix, dropout=self.dropout)

        # 4) "Concat" using a view and apply a final linear.
        node_hidden = node_hidden.transpose(1, 2).contiguous().view(n_batches, max_length, self.heads * self.d_k)
        edge_hidden = edge_hidden.permute(0, 2, 3, 1, 4).contiguous().view(n_batches, max_length, max_length, self.heads * self.d_k)
        
        return mish_function(self.linears[3](node_hidden)), mish_function(self.linears[4](edge_hidden))




class GraphTransformerDecoder(nn.Module):
    def __init__(self, encoder_decoder, prot_n_embed, prot_e_embed):
        super(GraphTransformerDecoder, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.prot_n_embed = prot_n_embed
        self.prot_e_embed = prot_e_embed
        #self.pos_embed = pos_embed
        

    def forward(self,edges_logits, nodes_logits, prot_n_features,prot_e_features,adj_matrix):
        """Take in and process src and target sequences."""
        # return self.predict(self.encode(src,  adj_matrix, edges_att))
        return self.decode(edges_logits, nodes_logits, prot_n_features, prot_e_features, adj_matrix)

    def decode(self,edges_logits, nodes_logits, prot_n_features, prot_e_features, adj_matrix):  # (batch, max_length, d_atom+1)
        # xv.shape = (batch, max_length, d_model)
 
        prot_n_initial = self.prot_n_embed(prot_n_features[:, :, :]) #+ self.pos_embed(prot_n_features[:, :, -1].squeeze(-1))
        # node_initial = self.node_embed(node_features[:, :, :-1])
        # evw = xv + evw for directions; evw.shape = (batch, max_length, max_length, d_model)
        # edge_initial = node_initial.unsqueeze(-2) + self.edge_embed(edge_features)

        prot_e_initial = self.prot_e_embed(prot_e_features)
   
        return self.encoder_decoder(edges_logits, nodes_logits,prot_n_initial, prot_e_initial, adj_matrix)



# Embeddings


class Node_Embeddings_dec(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings_dec, self).__init__()
        self.lut = nn.Linear(d_atom, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape(batch, max_length, d_atom)
      
        return self.dropout(self.lut(x.float())) * math.sqrt(self.d_emb)


class Edge_Embeddings_dec(nn.Module):
    def __init__(self, d_edge, d_emb, dropout):
        super(Edge_Embeddings_dec, self).__init__()
        
        self.lut = nn.Linear(d_edge, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape = (batch, max_length, max_length, d_edge)
    
        return self.dropout(self.lut(x.float())) * math.sqrt(self.d_emb)


class Position_Encoding_dec(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding_dec, self).__init__()
        self.max_length = max_length
        self.d_emb = d_emb
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Linear(max_length, d_emb*max_length)  
      
    def forward(self, x):
      
        return self.dropout(self.pe(x.float()).view(-1,self.max_length, self.d_emb))  # (batch, max_length) -> (batch, max_length, d_emb)


# Generator
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def swish_function(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish_function(x):
    return x * torch.tanh(F.softplus(x))


class Decoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, scale_norm):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self,edges_logits, nodes_logits, prot_n_initial, prot_e_initial, adj_matrix):
        """Pass the input through each layer in turn."""
        for layer in self.layers:
            edges_logits, nodes_logits, attn = layer(edges_logits, nodes_logits,prot_n_initial, prot_e_initial, adj_matrix)
            nodes_logits, edges_logits = self.norm(nodes_logits), self.norm(edges_logits)
            
   
        return edges_logits, nodes_logits, attn


class MoleculeEncoderDecoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm, dec_attn):
        super(MoleculeEncoderDecoderLayer, self).__init__()
        self.self_attn = self_attn        # MultiHeadedAttention
        self.feed_forward = feed_forward  # PositionwiseFeedForward
        # self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.dec_attn = dec_attn
        
    def forward(self, edges_hidden, nodes_hidden, prot_n, prot_e,adj_matrix):
        """Follow Figure 1 (left) for connections."""

        ########## PROTEIN ENCODING LAYER #############
        
        prot_n_hidden = self.dropout(self.norm(prot_n))
        prot_e_hidden = self.dropout(self.norm(prot_e))
        #prot_e_hidden_first, prot_n_hidden_first = self.self_attn(prot_n_hidden, prot_n_hidden, prot_e,adj_matrix)

        #prot_e_hidden_first = prot_e + self.dropout(self.norm(prot_e_hidden_first))
        
        #prot_n_hidden_second = self.feed_forward(prot_n_hidden_first)
        
        #prot_n_hidden_second = prot_n_hidden + self.dropout(self.norm(prot_n_hidden_first))
        
        #prot_e_hidden_second = self.feed_forward(prot_e_hidden_first)
     
        #prot_n_hidden_third =  prot_n_hidden_first + self.dropout(self.norm(prot_n_hidden_second))
        
        #prot_e_hidden_third = prot_e_hidden_first + self.dropout(self.norm(prot_e_hidden_second))
        
        ########## MOLECULE DECODING LAYER #############
        
        edge_hidden_first, node_hidden_first, attn = self.dec_attn(nodes_hidden,nodes_hidden, edges_hidden, adj_matrix,prot_n, prot_e)
        
        node_hidden_first = nodes_hidden + self.dropout(node_hidden_first)
        
        edge_hidden_first = edges_hidden + self.dropout(edge_hidden_first)
        
        node_hidden_second  = self.feed_forward(self.norm(node_hidden_first))
        
        edge_hidden_second =  self.feed_forward(self.norm(edge_hidden_first))
        
        
      

        return edge_hidden_second, node_hidden_second, attn


class PositionwiseFeedForwardDecoder(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.1, dense_output_nonlinearity='tanh'):
        super(PositionwiseFeedForwardDecoder, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'gelu':
            self.dense_output_nonlinearity = lambda x: F.gelu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
        elif dense_output_nonlinearity == 'swish':
            self.dense_output_nonlinearity = lambda x: x * torch.sigmoid(x)
        elif dense_output_nonlinearity == 'mish':
            self.dense_output_nonlinearity = lambda x: x * torch.tanh(F.softplus(x))

    def forward(self, node_hidden):
        if self.N_dense == 0:
            return node_hidden

        for i in range(self.N_dense - 1):
            node_hidden = self.dropout[i](mish_function(self.linears[i](node_hidden)))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](node_hidden)))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


# Attention
def dec_attention(query_mol_annot, key_prot_annot, value_mol_annot, query_mol_adj, key_prot_adj, value_mol_adj, adj_matrix, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # query_mol_annot.shape = (batch, h, max_length, d_e) 16,4,25,128
    # key_prot_annot.shape = (batch, h, max_length, d_e)  16,4,500,128
    # value_mol_annot.shape = (batch, h, max_length, d_e) 16,4,25,128
    
    # query_mol_adj.shape = (batch, h, max_length, max_length, d_e) 16,4,25,25,128
    # key_prot_adj.shape = (batch, h, max_length, max_length, d_e)  16,4,500,500,128
    # value_mol_adj.shape = (batch, h, max_length, max_length, d_e) 16,4,25,25,128
    
    
    
    # out_scores.shape = (batch, h, max_length, max_length) 16,4,25,25
    # in_scores.shape = (batch, h, max_length, max_length) 
    
    d_e = query_mol_annot.size(-1)
    
    out_scores = torch.einsum('bhmd,bhnd->bhmn', query_mol_annot, key_prot_annot) / math.sqrt(d_e) # 16, 4, 25, 128 ------ 16, 4, 500, 128
    in_scores = torch.einsum('bhmd,bhnd->bhnm', query_mol_annot, key_prot_annot) / math.sqrt(d_e)
    #out_scores = out_scores * adj_matrix.unsqueeze(1)
    
    out_attn = F.softmax(out_scores, dim=-1)
    in_attn = F.softmax(in_scores, dim=-1)
    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message = out_attn + in_attn.permute(0,1,3,2) #- diag_attn

    # add the diffusion caused by distance
    #message = message * adj_matrix.unsqueeze(1)

    if dropout is not None:
        message = dropout(message)

    node_hidden = torch.einsum('bhmn,bhmd->bhmd', message, value_mol_annot)

    out_scores_e = torch.einsum('bhmnd,bhkjd->bhmn', query_mol_adj, key_prot_adj) / math.sqrt(d_e)
    in_scores_e = torch.einsum('bhmnd,bhkjd->bhnm', query_mol_adj, key_prot_adj) / math.sqrt(d_e)
    out_attn_e = F.softmax(out_scores_e, dim=-1)
    in_attn_e = F.softmax(in_scores_e, dim=-1)
    diag_attn_e = torch.diag_embed(torch.diagonal(out_attn_e, dim1=-2, dim2=-1), dim1=-2, dim2=-1)    
    #out_scores_e = out_scores_e * adj_matrix.unsqueeze(1)
    
    message_e = out_attn_e + in_attn_e.permute(0,1,3,2) #- diag_attn_e

    edge_hidden = torch.einsum('bhmn,bhmkd->bhmnd', message_e, value_mol_adj)

    if dropout is not None:
        message_e = dropout(message_e)


    #score_prot = torch.einsum('bhmd,bhmnd->bhmn', key_prot_annot,key_prot_adj)

    #prot = torch.einsum('bhmn,bhnd->bhmd', score_prot, key_prot_annot)
 
    return edge_hidden, node_hidden, message


def prot_attention(query_prot_annot,key_prot_adj, value_prot_annot, adj_matrix, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # query.shape = (batch, h, max_length, d_e)
    # key.shape = (batch, h, max_length, max_length, d_e)
    # value.shape = (batch, h, max_length, d_e)
    # out_scores.shape = (batch, h, max_length, max_length)
    # in_scores.shape = (batch, h, max_length, max_length)

    d_e = query_prot_annot.size(-1)

        
    out_score_prot = torch.einsum('bhmd,bhmnd->bhmn', query_prot_annot,key_prot_adj)/ math.sqrt(d_e)
    in_score_prot = torch.einsum('bhmd,bhmnd->bhnm', query_prot_annot,key_prot_adj)/ math.sqrt(d_e)
 
    out_attn = F.softmax(out_score_prot, dim=-1)
    in_attn = F.softmax(in_score_prot, dim=-1)
    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message_e = out_attn + in_attn - diag_attn

        # add the diffusion caused by distance
       

    if dropout is not None:
        message_e = dropout(message_e)

    # message.shape = (batch, h, max_length, max_length), value.shape = (batch, h, max_length, d_k)
    prot = torch.einsum('bhmn,bhnd->bhmd', out_attn, value_prot_annot)
    prot_e = out_attn.unsqueeze(-1) * key_prot_adj

    return prot_e, prot, message_e


class MultiHeadedAttentionDecoder(nn.Module):
    def __init__(self, heads, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttentionDecoder, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads  # We assume d_v always equals d_k
        self.heads = heads

        self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))

        self.linears = clones(nn.Linear(d_model, d_model), 5)  # 5 for query, key, value, node update, edge update

        self.message = None
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

        
        self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
    

    def forward(self, query_node, value_node, key_edge, adj_matrix):
        """Implements Figure 2"""
        
        n_batches, max_length, d_model = query_node.shape

        # 1) Prepare adjacency matrix with shape (batch, max_length, max_length)
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        adj_matrix = self.attenuation_lambda * adj_matrix
        
        adj_matrix = self.distance_matrix_kernel(adj_matrix)

        # 2) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query_node).view(n_batches, max_length, self.heads, self.d_k).transpose(1, 2)
        key = self.linears[1](key_edge).view(n_batches, max_length, max_length, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        value = self.linears[2](value_node).view(n_batches, max_length, self.heads, self.d_k).transpose(1, 2)

        # 3) Apply attention on all the projected vectors in batch.
        prot_e_hidden, prot_n_hidden, self.message = prot_attention(query, key, value, adj_matrix, dropout=self.dropout)

        # 4) "Concat" using a view and apply a final linear.
        prot_n_hidden = prot_n_hidden.transpose(1, 2).contiguous().view(n_batches, max_length, self.heads * self.d_k)
        prot_e_hidden = prot_e_hidden.permute(0, 2, 3, 1, 4).contiguous().view(n_batches, max_length, max_length, self.heads * self.d_k)

        return mish_function(self.linears[3](prot_e_hidden)), mish_function(self.linears[4](prot_n_hidden))

class TransformerDecoderAttention(nn.Module):
    def __init__(self, heads, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1):
        """Take in model size and number of heads."""
        super(TransformerDecoderAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads  # We assume d_v always equals d_k
        self.heads = heads

        self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))

        self.linears = clones(nn.Linear(d_model, d_model), 8)  # 5 for query, key, value, node update, edge update

        self.message = None
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

        
        self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
    

    def forward(self, query_node, value_node, key_edge, adj_matrix, prot_n_hidden, prot_e):
        """Implements Figure 2"""
        
        n_batches, max_length_mol, d_model = query_node.shape
        max_length = prot_e.shape[1]
      
        # 1) Prepare adjacency matrix with shape (batch, max_length, max_length)
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        adj_matrix = self.attenuation_lambda * adj_matrix
        
        adj_matrix = self.distance_matrix_kernel(adj_matrix)

        # 2) Do all the linear projections in batch from d_model => h x d_k
        query_mol_annot = self.linears[0](query_node).view(n_batches, max_length_mol, self.heads, self.d_k).transpose(1, 2)
        key_prot_annot = self.linears[1](prot_n_hidden).view(1, max_length, self.heads, self.d_k).transpose(1, 2)
        value_mol_annot = self.linears[2](value_node).view(n_batches, max_length_mol, self.heads, self.d_k).transpose(1, 2)
 
        query_mol_adj = self.linears[3](key_edge).view(n_batches, max_length_mol, max_length_mol, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        key_prot_adj = self.linears[4](prot_e).view(1, max_length, max_length, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        value_mol_adj = self.linears[5](key_edge).view(n_batches, max_length_mol, max_length_mol, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
      
        # 3) Apply attention on all the projected vectors in batch.
        edge_hidden, node_hidden, self.message = dec_attention(query_mol_annot, key_prot_annot, value_mol_annot, query_mol_adj, key_prot_adj, value_mol_adj, adj_matrix, dropout=self.dropout)

        # 4) "Concat" using a view and apply a final linear.
        node_hidden = node_hidden.transpose(1, 2).contiguous().view(n_batches, max_length_mol, self.heads * self.d_k)
        edge_hidden = edge_hidden.permute(0, 2, 3, 1, 4).contiguous().view(n_batches, max_length_mol, max_length_mol, self.heads * self.d_k)
        
        return mish_function(self.linears[6](edge_hidden)), mish_function(self.linears[7](node_hidden)), self.message


