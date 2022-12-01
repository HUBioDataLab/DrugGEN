import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import Embedding,  ModuleList
from torch_geometric.nn import  PNAConv, global_add_pool, Set2Set, GraphMultisetTransformer
import math

class MLP(nn.Module):
    def __init__(self, act, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = torch.nn.ReLU()
        self.fc2 = nn.Linear(hid_feat,out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)

class Attention_new(nn.Module):
    def __init__(self, dim, heads, act, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = 1./dim**0.5


        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attention_dropout)


        self.d_k = dim // heads  
        self.heads = heads


       
        self.out_n = nn.Linear(dim, dim)
        self.out_e = nn.Linear(dim,dim)
    def forward(self, node, edge):
        b, n, c = node.shape
        
        
        q_embed = self.q(node).view(-1, n, self.heads, c//self.heads)
        k_embed = self.k(node).view(-1, n, self.heads, c//self.heads)
        v_embed = self.v(node).view(-1, n, self.heads, c//self.heads)
   
        e_embed = self.e(edge).view(-1, n, n, self.heads, c//self.heads)
        
        q_embed = q_embed.unsqueeze(2)
        k_embed = k_embed.unsqueeze(1)
        
        attn = q_embed * k_embed
        
        attn = attn/ math.sqrt(self.d_k)
        
     
        attn = attn * (e_embed + 1) * e_embed      

        e_embed = attn.flatten(3)
        
        edges = self.out_e(e_embed)
        
        attn = F.softmax(attn, dim=2)
        
        v_embed = v_embed.unsqueeze(1)
        
        v_embed = attn * v_embed
        
        v_embed = v_embed.sum(dim=2).flatten(2)
        
        nodes  = self.out_n(v_embed)
        

        
        return nodes, edges

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads,act, mlp_ratio=4, drop_rate=0., ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
   
        self.attn = Attention_new(dim, heads, act, drop_rate, drop_rate)
        self.ln3 = nn.LayerNorm(dim)

        self.mlp = MLP(act,dim,dim*mlp_ratio, dim, dropout=drop_rate)
 
        self.ln5 = nn.LayerNorm(dim)
      

    def forward(self, x,y):
        x1 = self.ln1(x)
        x2,attn = self.attn(x1,y)
        x2 = x1 + x2
        x2 = self.ln3(x2)     
        x = self.ln5(x2 + self.mlp(x2))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, act, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, act, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x,y):
        
        for Encoder_Block in self.Encoder_Blocks:
            x,  attn = Encoder_Block(x,y)
            
        return x, attn

class enc_dec_attention(nn.Module):
    def __init__(self, dim, heads, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = 1./dim**0.5
 
        
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
    
        self.dropout_dec = nn.Dropout(proj_dropout)
        self.out_nd = nn.Linear(dim, dim)
        self.out_ed = nn.Linear(dim,dim)
        
    def forward(self, mol_annot, drug_annot, mol_adj, drug_adj):
        
        b, n, c = mol_annot.shape
        _, m, _ = drug_annot.shape
     
        
        query_mol_annot = self.q_mx(drug_annot).view(-1,m, self.heads, c//self.heads)
        key_prot_annot = self.k_px(mol_annot).view(-1,n, self.heads, c//self.heads)
        value_mol_annot = self.v_mx(drug_annot).view(-1,m, self.heads, c//self.heads)
        
        drug_e = self.k_pa(drug_adj).view(-1,m,m, self.heads, c//self.heads)
        
        query_mol_annot = query_mol_annot.unsqueeze(2)
        key_prot_annot = key_prot_annot.unsqueeze(1)
        
        attn = query_mol_annot * key_prot_annot
        
        attn = attn/ math.sqrt(self.dim)

        
        attn = attn * (drug_e + 1) * drug_e      

        drug_e = attn.flatten(3)
        
        edges = self.out_ed(drug_e)
        
        attn = F.softmax(attn, dim=2)
        
        value_mol_annot = value_mol_annot.unsqueeze(1)
        
        value_mol_annot = attn * value_mol_annot
        
        value_mol_annot = value_mol_annot.sum(dim=2).flatten(2)
        
        nodes  = self.out_nd(value_mol_annot)          

        return nodes, edges

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

        self.mlp_ma = MLP(dim, dim, dropout=drop_rate)
        self.mlp_mx = MLP(dim, dim, dropout=drop_rate)
       
        self.ln4_ma = nn.LayerNorm(dim)
        self.ln4_mx = nn.LayerNorm(dim)
    
        
    def forward(self,mol_annot, prot_annot, mol_adj, prot_adj):

        mx = self.ln1_mx(mol_annot)
        px = self.ln1_px(prot_annot)
        
        ma = self.ln1_ma(mol_adj)
        pa = self.ln1_pa(prot_adj)
        
        px1, pa1= self.attn2(px, pa)
        
        px1 = px + px1
        pa1 = pa + pa1
        
        px1 = self.ln2_px(px1)
        pa1 = self.ln2_pa(pa1)
        
        mx1, ma1 = self.dec_attn(mx,px1,ma,pa1)
        
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
    
        return edge_hidden, node_hidden
    
class TransformerDecoder(nn.Module):
    def __init__(self, dim,  depth, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        
        self.Decoder_Blocks = nn.ModuleList([     
            Decoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])
        
    def forward(self, mol_annot, prot_annot, mol_adj, prot_adj):
        
        for Decoder_Block in self.Decoder_Blocks:
            edge_hidden, node_hidden = Decoder_Block(mol_annot, prot_annot, mol_adj, prot_adj)
            
        return edge_hidden, node_hidden



"""class PNA(torch.nn.Module):
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

        return  x"""




"""class GraphConvolution(nn.Module):

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

        return output"""

"""class Attention(nn.Module):
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
     
        return x, attn"""
    
    
    