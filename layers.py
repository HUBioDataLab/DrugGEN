import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.act = torch.nn.ReLU()
        self.fc2 = nn.Linear(hid_feat,out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)

class Attention_new(nn.Module):
    def __init__(self, dim, heads, attention_dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = 1./dim**0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        #self.attention_dropout = nn.Dropout(attention_dropout)

        self.d_k = dim // heads  
        self.heads = heads
        self.out_e = nn.Linear(dim,dim)
        self.out_n = nn.Linear(dim, dim)
        
        
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

        edge = self.out_e(attn.flatten(3))  
      
        attn = F.softmax(attn, dim=2)
        
        v_embed = v_embed.unsqueeze(1)
  
        v_embed = attn * v_embed
        
        v_embed = v_embed.sum(dim=2).flatten(2)
        
        node  = self.out_n(v_embed)
           
        return node, edge

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads,act, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
   
        self.attn = Attention_new(dim, heads, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dim, dropout=drop_rate)
        self.mlp2 = MLP(dim, dim*mlp_ratio, dim, dropout=drop_rate)
        self.ln5 = nn.LayerNorm(dim)
        self.ln6 = nn.LayerNorm(dim)

    def forward(self, x,y):
        x1 = self.ln1(x)
        x2,y1 = self.attn(x1,y)
        x2 = x1 + x2
        y2 = y1 + y
        x2 = self.ln3(x2)   
        y2 = self.ln4(y2)   
        
        x = self.ln5(x2 + self.mlp(x2))
        y = self.ln6(y2 + self.mlp2(y2))
        return x, y


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, act, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, act, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x,y):
        
        for Encoder_Block in self.Encoder_Blocks:
            x,  y = Encoder_Block(x,y)
            
        return x, y

class enc_dec_attention(nn.Module):
    def __init__(self, dim, heads, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = 1./dim**0.5
 
        
        "query is molecules"
        "key is prot"
        "values is again molecule"
        self.q_mx = nn.Linear(dim,dim)
        self.k_px = nn.Linear(dim,dim)
        self.v_mx = nn.Linear(dim,dim)
        
        
        self.k_pa = nn.Linear(dim,dim)
        self.v_ma = nn.Linear(dim,dim)
        
    

 
    
        #self.dropout_dec = nn.Dropout(proj_dropout)
        self.out_nd = nn.Linear(dim, dim)
        self.out_ed = nn.Linear(dim,dim)
        
    def forward(self, mol_annot, prot_annot, mol_adj, prot_adj):
        
        b, n, c = mol_annot.shape
        _, m, _ = prot_annot.shape
     
        
        query_mol_annot = self.q_mx(mol_annot).view(-1,m, self.heads, c//self.heads)
        key_prot_annot = self.k_px(prot_annot).view(-1,n, self.heads, c//self.heads)
        value_mol_annot = self.v_mx(mol_annot).view(-1,m, self.heads, c//self.heads)
        
        mol_e = self.v_ma(mol_adj).view(-1,m,m, self.heads, c//self.heads)
        prot_e = self.k_pa(prot_adj).view(-1,m,m, self.heads, c//self.heads)
        
        query_mol_annot = query_mol_annot.unsqueeze(2)
        key_prot_annot = key_prot_annot.unsqueeze(1)
        
        
        
        #attn = torch.einsum('bnchd,bmahd->bnahd', query_mol_annot, key_prot_annot)
        
        attn = query_mol_annot * key_prot_annot
        
        attn = attn/ math.sqrt(self.dim)

        
        attn = attn * (prot_e + 1) * mol_e      
      
        mol_e_new = attn.flatten(3)
        
        mol_adj = self.out_ed(mol_e_new)
        
        attn = F.softmax(attn, dim=2)
        
        value_mol_annot = value_mol_annot.unsqueeze(1)
        
        value_mol_annot = attn * value_mol_annot
        
        value_mol_annot = value_mol_annot.sum(dim=2).flatten(2)
        
        mol_annot  = self.out_nd(value_mol_annot)          

        return mol_annot, prot_annot, mol_adj, prot_adj

class Decoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        
        self.ln1_ma = nn.LayerNorm(dim)
        self.ln1_pa = nn.LayerNorm(dim)
        self.ln1_mx = nn.LayerNorm(dim)
        self.ln1_px = nn.LayerNorm(dim)
        
        self.attn2 = Attention_new(dim, heads, drop_rate)
        
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

        mol_annot = self.ln1_mx(mol_annot)
        mol_adj = self.ln1_ma(mol_adj)
        
        prot_annot = self.ln1_px(prot_annot)
        prot_adj = self.ln1_pa(prot_adj)
        
        px1, pa1= self.attn2(prot_annot, prot_adj)
        
        prot_annot = prot_annot + px1
        prot_adj = prot_adj + pa1
        
        prot_annot = self.ln2_px(prot_annot)
        prot_adj = self.ln2_pa(prot_adj)
        
        mx1, prot_annot, ma1, prot_adj  = self.dec_attn(mol_annot,prot_annot,mol_adj,prot_adj)
        
        ma1 = mol_adj + ma1
        mx1 = mol_annot + mx1
        
        ma2 = self.ln3_ma(ma1)
        mx2 = self.ln3_mx(mx1)
        
        ma3 = self.mlp_ma(ma2)
        mx3 = self.mlp_mx(mx2)
        
        ma = ma3 + ma2
        mx = mx3 + mx2
        
        mol_adj = self.ln4_ma(ma)
        mol_annot = self.ln4_mx(mx)           
    
        return mol_annot, prot_annot, mol_adj, prot_adj
    
class TransformerDecoder(nn.Module):
    def __init__(self, dim,  depth, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        
        self.Decoder_Blocks = nn.ModuleList([     
            Decoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])
        
    def forward(self, mol_annot, prot_annot, mol_adj, prot_adj):
        
        for Decoder_Block in self.Decoder_Blocks:
            mol_annot, prot_annot, mol_adj, prot_adj  = Decoder_Block(mol_annot, prot_annot, mol_adj, prot_adj)
            
        return mol_annot, prot_annot,mol_adj, prot_adj
    