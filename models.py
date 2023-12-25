import torch
import torch.nn as nn
from layers import TransformerEncoder

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, act, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio, submodel):
        super(Generator, self).__init__()
        self.submodel = submodel
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()

        self.features = vertexes * vertexes * edges + vertexes * nodes
        self.transformer_dim = vertexes * vertexes * dim + vertexes * dim
        self.pos_enc_dim = 5

        self.node_layers = nn.Sequential(nn.Linear(nodes, 64), act, nn.Linear(64,dim), act, nn.Dropout(self.dropout))
        self.edge_layers = nn.Sequential(nn.Linear(edges, 64), act, nn.Linear(64,dim), act, nn.Dropout(self.dropout))
        self.TransformerEncoder = TransformerEncoder(dim=self.dim, depth=self.depth, heads=self.heads, act = act,
                                                                    mlp_ratio=self.mlp_ratio, drop_rate=self.dropout)

        self.readout_e = nn.Linear(self.dim, edges)
        self.readout_n = nn.Linear(self.dim, nodes)
        self.softmax = nn.Softmax(dim = -1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def laplacian_positional_enc(self, adj):
        A = adj
        D = torch.diag(torch.count_nonzero(A, dim=-1))
        L = torch.eye(A.shape[0], device=A.device) - D * A * D

        EigVal, EigVec = torch.linalg.eig(L)
        idx = torch.argsort(torch.real(EigVal))
        EigVal, EigVec = EigVal[idx], torch.real(EigVec[:,idx])
        pos_enc = EigVec[:,1:self.pos_enc_dim + 1]
        return pos_enc

    def forward(self, z_e, z_n):
        b, n, c = z_n.shape
        _, _, _ , d = z_e.shape

        node = self.node_layers(z_n)
        edge = self.edge_layers(z_e)
        edge = (edge + edge.permute(0, 2, 1, 3)) / 2

        node, edge = self.TransformerEncoder(node,edge)

        node_sample = self.readout_n(node)
        edge_sample = self.readout_e(edge)
        return node, edge, node_sample, edge_sample


class simple_disc(nn.Module):
    def __init__(self, act, m_dim, vertexes, b_dim):
        super().__init__()

        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()

        features = vertexes * m_dim + vertexes * vertexes * b_dim
        self.predictor = nn.Sequential(nn.Linear(features,256), act, nn.Linear(256,128), act, nn.Linear(128,64), act,
                                       nn.Linear(64,32), act, nn.Linear(32,16), act,
                                       nn.Linear(16,1))

    def forward(self, x):
        prediction = self.predictor(x)
        return prediction
