import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
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

        node = self.out_n(v_embed)
        return node, edge


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, act, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention_new(dim, heads, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dim, dropout=drop_rate)
        self.mlp2 = MLP(dim, dim*mlp_ratio, dim, dropout=drop_rate)
        self.ln5 = nn.LayerNorm(dim)
        self.ln6 = nn.LayerNorm(dim)

    def forward(self, x, y):
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

    def forward(self, x, y):
        for Encoder_Block in self.Encoder_Blocks:
            x, y = Encoder_Block(x,y)
        return x, y
