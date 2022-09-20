import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import Embedding,  ModuleList
from torch_geometric.nn import  PNAConv, global_add_pool, Set2Set, GraphMultisetTransformer
import math

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)

class Attention_new(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = 1./dim**0.5
        #self.scale = torch.div(1, torch.pow(dim, 0.5)) #1./torch.pow(dim, 0.5) #dim**0.5 torch.div(x, 0.5)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.attention_dropout = nn.Dropout(attention_dropout)

        #self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
        self.d_k = dim // heads  # We assume d_v always equals d_k
        self.heads = heads


        self.out_e = nn.Linear(dim,dim)
        self.out_n = nn.Linear(dim,dim)
        
    def forward(self, node, edge):
        b, n, c = node.shape
        b1, n1, n2, c1 = edge.shape
        
        q_embed = self.q(node).view(-1, self.heads, n, c//self.heads)
        k_embed = self.k(edge).view(-1, self.heads, n1, n2, c1//self.heads)
        v_embed = self.v(node).view(-1, self.heads, n, c//self.heads)
        #x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        


        out_scores = torch.einsum('bhmd,bhmnd->bhmn', q_embed, k_embed) / math.sqrt(self.scale)
        in_scores = torch.einsum('bhmd,bhmnd->bhnm', q_embed, k_embed) / math.sqrt(self.scale)

        out_attn = F.softmax(out_scores, dim=-1)
        in_attn = F.softmax(in_scores, dim=-1)
        diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

        message = out_attn + in_attn - diag_attn
        
        # add the diffusion caused by distance
    
        #message = message * adj_matrix.unsqueeze(1)

        #if dropout is not None:
        message = self.attention_dropout(message)

        # message.shape = (batch, h, max_length, max_length), value.shape = (batch, h, max_length, d_k)
        node_hidden = torch.einsum('bhmn,bhnd->bhmd', message, v_embed)
        edge_hidden = torch.einsum('bhmn,bhand->bhamd', message, k_embed)
    
 
        #edge_hidden = message.unsqueeze(-1) * k_embed
        
        node_hidden = self.out_n(node_hidden.view(b, n, c))
        edge_hidden = self.out_e(edge_hidden.reshape(b1, n1, n2, c1))
        
        return node_hidden, edge_hidden, message

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = Attention_new(dim, heads, drop_rate, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)
        self.mlp2 = MLP(dim, dim*mlp_ratio, dropout=drop_rate)
        self.ln5 = nn.LayerNorm(dim)
        self.ln6 = nn.LayerNorm(dim)       

    def forward(self, x, y):
        x1 = self.ln1(x)
        y1 = self.ln2(y)
        x2, y2, attn = self.attn(x1, y1)
        x2 = x1 + x2
        y2 = y1 + y2
        x2 = self.ln3(x2)
        y2 = self.ln4(y2)
        x = self.ln5(x2 + self.mlp(x2))
        y = self.ln6(y2 + self.mlp2(y2))
        return x, y, attn


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x, y):
        
        for Encoder_Block in self.Encoder_Blocks:
            x, y, attn = Encoder_Block(x, y)
            
        return x, y, attn

class enc_dec_attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()

        self.heads = heads
        self.scale = 1./dim**0.5
        #self.scale = torch.div(1, torch.pow(dim, 0.5)) #1./torch.pow(dim, 0.5) #dim**0.5 torch.div(x, 0.5)
        
        "query is molecules"
        "key is protein"
        "values is again molecule"
        self.q_mx = nn.Linear(dim,dim)
        self.k_px = nn.Linear(dim,dim)
        self.v_mx = nn.Linear(dim,dim)
        
        self.q_ma = nn.Linear(dim,dim)
        self.k_pa = nn.Linear(dim,dim)
        self.v_ma = nn.Linear(dim,dim)
        
    
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_mx = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )
        self.out_ma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )        
        #self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
        self.dropout_dec = nn.Dropout(proj_dropout)

    def forward(self, mol_annot, prot_annot, mol_adj, prot_adj):
        # query_mol_annot.shape = (batch, h, max_length, d_e) 16,4,25,128
        # key_prot_annot.shape = (batch, h, max_length, d_e)  16,4,500,128
        # value_mol_annot.shape = (batch, h, max_length, d_e) 16,4,25,128
        
        # query_mol_adj.shape = (batch, h, max_length, max_length, d_e) 16,4,25,25,128
        # key_prot_adj.shape = (batch, h, max_length, max_length, d_e)  16,4,500,500,128
        # value_mol_adj.shape = (batch, h, max_length, max_length, d_e) 16,4,25,25,128
        
        # out_scores.shape = (batch, h, max_length, max_length) 16,4,25,25
        # in_scores.shape = (batch, h, max_length, max_length) 
        
        b, n, c = mol_annot.shape
        bp, np, cp = prot_annot.shape
        
        b1, n1, n2, c1 = mol_adj.shape
        bpa, npa, npa2, cpa = prot_adj.shape
        
        query_mol_annot = self.q_mx(mol_annot).view(-1,self.heads, n, c//self.heads)
        key_prot_annot = self.k_px(prot_annot).view(-1,self.heads, np, cp//self.heads)
        value_mol_annot = self.v_mx(mol_annot).view(-1,self.heads, n, c//self.heads)
        
        query_mol_adj = self.q_ma(mol_adj).view(-1, self.heads, n1, n2, c1//self.heads)
        key_prot_adj = self.k_pa(prot_adj).view(-1, self.heads, npa, npa2, cpa//self.heads)
        value_mol_adj = self.v_ma(mol_adj).view(-1, self.heads, n1, n2, c1//self.heads)
        
        
        out_scores = torch.einsum('bhmd,bhnd->bhmn', query_mol_annot, key_prot_annot) / math.sqrt(self.scale) # 16, 4, 25, 128 ------ 16, 4, 500, 128
        in_scores = torch.einsum('bhmd,bhnd->bhnm', query_mol_annot, key_prot_annot) / math.sqrt(self.scale)
        #out_scores = out_scores * adj_matrix.unsqueeze(1)
        
        out_attn = F.softmax(out_scores, dim=-1)
        in_attn = F.softmax(in_scores, dim=-1)
        #diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

        message = out_attn + in_attn.permute(0,1,3,2) #- diag_attn

        # add the diffusion caused by distance
        #message = message * adj_matrix.unsqueeze(1)

        message = self.dropout_dec(message)

        node_hidden = torch.einsum('bhmn,bhmd->bhmd', message, value_mol_annot)


        out_scores_e = torch.einsum('bhmnd,bhkjd->bhmn', query_mol_adj, key_prot_adj) / math.sqrt(self.scale)
        in_scores_e = torch.einsum('bhmnd,bhkjd->bhnm', query_mol_adj, key_prot_adj) / math.sqrt(self.scale)
        
        out_attn_e = F.softmax(out_scores_e, dim=-1)
        in_attn_e = F.softmax(in_scores_e, dim=-1)
        #diag_attn_e = torch.diag_embed(torch.diagonal(out_attn_e, dim1=-2, dim2=-1), dim1=-2, dim2=-1)    
        #out_scores_e = out_scores_e * adj_matrix.unsqueeze(1)
        
        message_e = out_attn_e + in_attn_e.permute(0,1,3,2) #- diag_attn_e

        edge_hidden = torch.einsum('bhmn,bhmkd->bhmnd', message_e, value_mol_adj)

        message_e = self.dropout_dec(message_e)

        node_hidden = self.out_mx(node_hidden.view(b, n, c))
        edge_hidden = self.out_ma(edge_hidden.reshape(b1, n1, n2, c1))        

        return edge_hidden, node_hidden, message

class Decoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        
        self.ln1_ma = nn.LayerNorm(dim)
        self.ln1_pa = nn.LayerNorm(dim)
        self.ln1_mx = nn.LayerNorm(dim)
        self.ln1_px = nn.LayerNorm(dim)
        
        self.attn2 = Attention_new(dim, heads, drop_rate, drop_rate)
        
        self.ln2_pa = nn.LayerNorm(dim)
        self.ln2_px = nn.LayerNorm(dim)
        
        self.dec_attn = enc_dec_attention(dim, heads, drop_rate, drop_rate)
        
        self.ln3_ma = nn.LayerNorm(dim)
        self.ln3_mx = nn.LayerNorm(dim)

        self.mlp_ma = MLP(dim, dim*mlp_ratio, dropout=drop_rate)
        self.mlp_mx = MLP(dim, dim*mlp_ratio, dropout=drop_rate)
       
        self.ln4_ma = nn.LayerNorm(dim)
        self.ln4_mx = nn.LayerNorm(dim)
    
        
    def forward(self,mol_annot, prot_annot, mol_adj, prot_adj):

        mx = self.ln1_mx(mol_annot)
        px = self.ln1_px(prot_annot)
        
        ma = self.ln1_ma(mol_adj)
        pa = self.ln1_pa(prot_adj)
        
        px1, pa1, prot_attn = self.attn2(px, pa)
        
        px1 = px + px1
        pa1 = pa + pa1
        
        px1 = self.ln2_px(px1)
        pa1 = self.ln2_pa(pa1)
        
        ma1, mx1, attn_dec = self.dec_attn(mx,px1,ma,pa1)
        
        ma1 = ma + ma1
        mx1 = mx + mx1
        
        ma1 = self.ln3_ma(ma1)
        mx1 = self.ln3_mx(mx1)
        
        ma2 = self.mlp_ma(ma1)
        mx2 = self.mlp_mx(mx1)
        
        ma2 = ma2 + ma1
        mx2 = mx2 + mx1
        
        edge_hidden = self.ln4_ma(ma2)
        node_hidden = self.ln4_mx(mx2)        
    
        return edge_hidden, node_hidden, attn_dec
    
class TransformerDecoder(nn.Module):
    def __init__(self, dim,  depth, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        
        self.Decoder_Blocks = nn.ModuleList([     
            Decoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])
        
    def forward(self, mol_annot, prot_annot, mol_adj, prot_adj):
        
        for Decoder_Block in self.Decoder_Blocks:
            edge_hidden, node_hidden, message = Decoder_Block(mol_annot, prot_annot, mol_adj, prot_adj)
            
        return edge_hidden, node_hidden, message



class PNA(torch.nn.Module):
    def __init__(self,deg,agg,sca,pna_in_ch,pna_out_ch,edge_dim,towers,pre_lay,post_lay,pna_layer_num, graph_add):
        super(PNA,self).__init__()
                                                                 
        self.node_emb = Embedding(30, pna_in_ch)
        self.edge_emb = Embedding(30, edge_dim)
        degree = deg
        aggregators =   agg.split(",") #["max"]    #   'sum', 'min', 'max' 'std', 'var' 'mean',                                    ## buraları değiştirerek bak.
        scalers =  sca.split(",")   # ['amplification', 'attenuation']   #  'amplification', 'attenuation' , 'linear', 'inverse_linear, 'identity'
        self.graph_add = graph_add
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(pna_layer_num):                                               ##### layer sayısını hyperparameter olarak ayarla??   
            conv = PNAConv(in_channels=pna_in_ch, out_channels=pna_out_ch,
                           aggregators=aggregators, scalers=scalers, deg=degree,
                           edge_dim=edge_dim, towers=towers, pre_layers=pre_lay, post_layers=post_lay,  ## tower sayısını değiştirerek dene, default - 1
                           divide_input=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.LayerNorm(pna_out_ch))
        
        #self.graph_multitrans = GraphMultisetTransformer(in_channels=pna_out_ch, hidden_channels= 200, 
                                                         #out_channels= pna_out_ch, layer_norm = True)
        if self.graph_add == "set2set":
            self.s2s = Set2Set(in_channels=pna_out_ch, processing_steps=1, num_layers=1)

        if self.graph_add == "set2set":
            pna_out_ch = pna_out_ch*2
        self.mlp = nn.Sequential(nn.Linear(pna_out_ch,pna_out_ch), nn.Tanh(), nn.Linear(pna_out_ch,25), nn.Tanh(),nn.Linear(25,1))
        
    def forward(self, x, edge_index, edge_attr, batch):

        x = self.node_emb(x.squeeze())

        edge_attr = self.edge_emb(edge_attr)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
 
        if self.graph_add == "global_add":
            x = global_add_pool(x, batch.squeeze())
   
        elif self.graph_add == "set2set":

            x = self.s2s(x, batch.squeeze())
        #elif self.graph_add == "graph_multitrans":
            #x = self.graph_multitrans(x,batch.squeeze(),edge_index)
        x = self.mlp(x)

        return  x




class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout,gcn_depth):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        
        self.gcn_depth = gcn_depth
        
        self.out_feature_list = out_feature_list
                
        self.gcn_in = nn.Sequential(nn.Linear(in_features,out_feature_list[0]),nn.Tanh(),
                                     nn.Linear(out_feature_list[0],out_feature_list[0]),nn.Tanh(), 
                                     nn.Linear(out_feature_list[0], out_feature_list[0]), nn.Dropout(dropout))      
        
        self.gcn_convs = nn.ModuleList()
  
        for _ in range(gcn_depth):  
                  
                gcn_conv = nn.Sequential(nn.Linear(out_feature_list[0],out_feature_list[0]),nn.Tanh(),
                                         nn.Linear(out_feature_list[0],out_feature_list[0]),nn.Tanh(),
                                         nn.Linear(out_feature_list[0], out_feature_list[0]), nn.Dropout(dropout))
                
                self.gcn_convs.append(gcn_conv)
                
        self.gcn_out = nn.Sequential(nn.Linear(out_feature_list[0],out_feature_list[0]),nn.Tanh(),
                                     nn.Linear(out_feature_list[0],out_feature_list[0]),nn.Tanh(), 
                                     nn.Linear(out_feature_list[0], out_feature_list[1]), nn.Dropout(dropout))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9
        hidden = torch.stack([self.gcn_in(input) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
    
        hidden = torch.sum(hidden, 1) + self.gcn_in(input)
        hidden = activation(hidden) if activation is not None else hidden             

        for gcn_conv in self.gcn_convs:
            hidden1 = torch.stack([gcn_conv(hidden) for _ in range(adj.size(1))], 1)
            hidden1 = torch.einsum('bijk,bikl->bijl', (adj, hidden1))
            hidden = torch.sum(hidden1, 1) + gcn_conv(hidden)
            hidden = activation(hidden) if activation is not None else hidden
      
        output = torch.stack([self.gcn_out(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.gcn_out(hidden)
        output = activation(output) if activation is not None else output
        
        
        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, m_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+m_dim, out_features), nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+m_dim, out_features), nn.Tanh())                                    
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output

class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5
        #self.scale = torch.div(1, torch.pow(dim, 0.5)) #1./torch.pow(dim, 0.5) #dim**0.5 torch.div(x, 0.5)

        self.qkv = nn.Linear(dim, dim*3, bias=False)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )
        #self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
    
    def forward(self, x):
        b, n, c = x.shape

        #x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)

        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale

        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)
      

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        x = self.out(x)
     
        return x, attn
    
    
    