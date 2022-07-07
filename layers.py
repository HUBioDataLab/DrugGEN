import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import Embedding,  ModuleList
from torch_geometric.nn import  PNAConv, global_add_pool, Set2Set, GraphMultisetTransformer


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

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x2, attn = self.attn(x1)
        x2 = x1 + x2
        x2 = self.ln2(x2)
        x = x2 + self.mlp(x2)
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x, attn = Encoder_Block(x)
        return x, attn

class enc_dec_attention(nn.Module):
    def __init__(self, dim, x_molecule, x_protein, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.molecules = x_molecule
        self.protein = x_protein
        self.heads = heads
        self.scale = 1./dim**0.5
        #self.scale = torch.div(1, torch.pow(dim, 0.5)) #1./torch.pow(dim, 0.5) #dim**0.5 torch.div(x, 0.5)
        
        "query is molecules"
        "key is protein"
        "values is again molecule"
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))


    def forward(self, x, y):
        x = self.molecules
        y = self.protein

        b, n, c = x.shape
        batch, n_dim, channels = y.shape

        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, v = qv.permute(2, 0, 3, 1, 4)

        y = y + torch.randn([y.size(0), y.size(1), 1], device=y.device) * self.noise_strength_1
        k = self.qkv(y).reshape(batch, n_dim, 3, self.heads, channels//self.heads)
        k = k.permute(2, 0, 3, 1, 4)


        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)


       
        z = (attn @ v).transpose(1, 2).reshape(b, n, c)

        
        z = self.out(z)

       
        return z

class Decoder_Block(nn.Module):
    def __init__(self, dim, x_molecule, x_protein, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.molecule = x_molecule
        self.protein = x_protein
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.dec_attn = enc_dec_attention(dim, x_molecule, x_protein, heads, drop_rate, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)
    def forward(self,prot):
        x1= self.ln1(self.molecule)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.dec_attn(x2,prot)
        x3 = self.ln3(x)
        x = x + self.mlp(x3)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, dim, x_molecule, x_protein, depth, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Decoder_Blocks = nn.ModuleList([
            Decoder_Block(dim, x_molecule, x_protein, heads, mlp_ratio, drop_rate)
            for i in range(depth)])
    def forward(self, x,prot):
        for Decoder_Block in self.Decoder_Blocks:
            x = Decoder_Block(x,prot)
        return x



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




class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        
        self.out_feature_list = out_feature_list
        
        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9
        
        
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden

        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)
        
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
