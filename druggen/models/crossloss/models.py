from torch import nn
import torch
from ...layers import TransformerEncoder, TransformerDecoder


class CrossLossGenerator(nn.Module):
    """Generator network."""
    def __init__(self, config):
        self.config = config
        super().__init__()

        if config.act == "relu":
            act = nn.ReLU()
        elif config.act == "leaky":
            act = nn.LeakyReLU()
        elif config.act == "sigmoid":
            act = nn.Sigmoid()
        elif config.act == "tanh":
            act = nn.Tanh()

        self.features = config.vertexes * config.vertexes * config.edges + config.vertexes * config.nodes
        self.transformer_dim = config.vertexes * config.vertexes * config.dim + config.vertexes * config.dim
        self.pos_enc_dim = 5
        #self.pos_enc = nn.Linear(self.pos_enc_dim, self.dim)
        
        self.node_layers = nn.Sequential(nn.Linear(config.nodes, 64), act, nn.Linear(64,config.dim), act, nn.Dropout(config.dropout))
        self.edge_layers = nn.Sequential(nn.Linear(config.edges, 64), act, nn.Linear(64,config.dim), act, nn.Dropout(config.dropout))
        
        self.TransformerEncoder = TransformerEncoder(dim=config.dim, depth=config.depth, heads=config.heads, act = act,
                                                                    mlp_ratio=config.mlp_ratio, drop_rate=config.dropout)         

        self.readout_e = nn.Linear(config.dim, config.edges)
        self.readout_n = nn.Linear(config.dim, config.nodes)
        self.softmax = nn.Softmax(dim = -1)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # def laplacian_positional_enc(self, adj):
        
    #     A = adj
    #     D = torch.diag(torch.count_nonzero(A, dim=-1))
    #     L = torch.eye(A.shape[0], device=A.device) - D * A * D
        
    #     EigVal, EigVec = torch.linalg.eig(L)
    
    #     idx = torch.argsort(torch.real(EigVal))
    #     EigVal, EigVec = EigVal[idx], torch.real(EigVec[:,idx])
    #     pos_enc = EigVec[:,1:self.pos_enc_dim + 1]
        
    #     return pos_enc

    def forward(self, z_e, z_n):
        b, n, c = z_n.shape
        _, _, _ , d = z_e.shape
        #random_mask_e = torch.randint(low=0,high=2,size=(b,n,n,d)).to(z_e.device).float()
        #random_mask_n = torch.randint(low=0,high=2,size=(b,n,c)).to(z_n.device).float()
        #z_e = F.relu(z_e - random_mask_e)
        #z_n = F.relu(z_n - random_mask_n)

        #mask = self._generate_square_subsequent_mask(self.vertexes).to(z_e.device)
        
        node = self.node_layers(z_n)
        
        edge = self.edge_layers(z_e)
        
        edge = (edge + edge.permute(0,2,1,3))/2
        
        #lap = [self.laplacian_positional_enc(torch.max(x,-1)[1]) for x in edge]
        
        #lap = torch.stack(lap).to(node.device)
        
        #pos_enc = self.pos_enc(lap)
        
        #node = node + pos_enc
        
        node, edge = self.TransformerEncoder(node,edge)

        node_sample = self.softmax(self.readout_n(node))
        
        edge_sample = self.softmax(self.readout_e(edge))
        
        return node, edge, node_sample, edge_sample
     

class CrossLossDiscriminator(nn.Module):
    # def __init__(self, act, m_dim, vertexes, b_dim):
    def __init__(self, config):
        self.config = config
        super().__init__()
        if config.act == "relu":
            act = nn.ReLU()
        elif config.act == "leaky":
            act = nn.LeakyReLU()
        elif config.act == "sigmoid":
            act = nn.Sigmoid()
        elif config.act == "tanh":
            act = nn.Tanh()  
        features = config.vertexes * config.nodes + config.vertexes * config.vertexes * config.edges 
        
        self.predictor = nn.Sequential(
            nn.Linear(features,256),
            act,
            nn.Linear(256,128),
            act,
            nn.Linear(128,64),
            act,
            nn.Linear(64,32),
            act,
            nn.Linear(32,16),
            act,
            nn.Linear(16,1)
        )
    
    def forward(self, x):
        prediction = self.predictor(x)
        #prediction = F.softmax(prediction,dim=-1)
        return prediction
