import torch
import torch.nn as nn
from src.model.layers import TransformerEncoder

class Generator(nn.Module):
    """
    Generator network that uses a Transformer Encoder to process node and edge features.
    
    The network first processes input node and edge features with separate linear layers,
    then applies a Transformer Encoder to model interactions, and finally outputs both transformed
    features and readout samples.
    """
    def __init__(self, act, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio):
        """
        Initializes the Generator.

        Args:
            act (str): Type of activation function to use ("relu", "leaky", "sigmoid", or "tanh").
            vertexes (int): Number of vertexes in the graph.
            edges (int): Number of edge features.
            nodes (int): Number of node features.
            dropout (float): Dropout rate.
            dim (int): Dimensionality used for intermediate features.
            depth (int): Number of Transformer encoder blocks.
            heads (int): Number of attention heads in the Transformer.
            mlp_ratio (int): Ratio for determining hidden layer size in MLP modules.
        """
        super(Generator, self).__init__()
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Set the activation function based on the provided string
        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()

        # Calculate the total number of features and dimensions for transformer
        self.features = vertexes * vertexes * edges + vertexes * nodes
        self.transformer_dim = vertexes * vertexes * dim + vertexes * dim

        self.node_layers = nn.Sequential(
            nn.Linear(nodes, 64), act,
            nn.Linear(64, dim), act,
            nn.Dropout(self.dropout)
        )
        self.edge_layers = nn.Sequential(
            nn.Linear(edges, 64), act,
            nn.Linear(64, dim), act,
            nn.Dropout(self.dropout)
        )
        self.TransformerEncoder = TransformerEncoder(
            dim=self.dim, depth=self.depth, heads=self.heads, act=act,
            mlp_ratio=self.mlp_ratio, drop_rate=self.dropout
        )

        self.readout_e = nn.Linear(self.dim, edges)
        self.readout_n = nn.Linear(self.dim, nodes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z_e, z_n):
        """
        Forward pass of the Generator.
        
        Args:
            z_e (torch.Tensor): Edge features tensor of shape (batch, vertexes, vertexes, edges).
            z_n (torch.Tensor): Node features tensor of shape (batch, vertexes, nodes).
        
        Returns:
            tuple: A tuple containing:
                - node: Updated node features after the transformer.
                - edge: Updated edge features after the transformer.
                - node_sample: Readout sample from node features.
                - edge_sample: Readout sample from edge features.
        """
        b, n, c = z_n.shape
        # The fourth dimension of edge features
        _, _, _, d = z_e.shape

        # Process node and edge features through their respective layers
        node = self.node_layers(z_n)
        edge = self.edge_layers(z_e)
        # Symmetrize the edge features by averaging with its transpose along vertex dimensions
        edge = (edge + edge.permute(0, 2, 1, 3)) / 2

        # Pass the features through the Transformer Encoder
        node, edge = self.TransformerEncoder(node, edge)

        # Readout layers to generate final outputs
        node_sample = self.readout_n(node)
        edge_sample = self.readout_e(edge)

        return node, edge, node_sample, edge_sample


class Discriminator(nn.Module):
    """
    Discriminator network that evaluates node and edge features.
    
    It processes features with linear layers, applies a Transformer Encoder to capture dependencies,
    and finally predicts a scalar value using an MLP on aggregated node features.

    This class is used in DrugGEN model.
    """
    def __init__(self, act, vertexes, edges, nodes, dropout, dim, depth, heads, mlp_ratio):
        """
        Initializes the Discriminator.

        Args:
            act (str): Activation function type ("relu", "leaky", "sigmoid", or "tanh").
            vertexes (int): Number of vertexes.
            edges (int): Number of edge features.
            nodes (int): Number of node features.
            dropout (float): Dropout rate.
            dim (int): Dimensionality for intermediate representations.
            depth (int): Number of Transformer encoder blocks.
            heads (int): Number of attention heads.
            mlp_ratio (int): MLP ratio for hidden layer dimensions.
        """
        super(Discriminator, self).__init__()
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Set the activation function
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

        # Define layers for processing node and edge features
        self.node_layers = nn.Sequential(
            nn.Linear(nodes, 64), act,
            nn.Linear(64, dim), act,
            nn.Dropout(self.dropout)
        )
        self.edge_layers = nn.Sequential(
            nn.Linear(edges, 64), act,
            nn.Linear(64, dim), act,
            nn.Dropout(self.dropout)
        )
        # Transformer Encoder for modeling node and edge interactions
        self.TransformerEncoder = TransformerEncoder(
            dim=self.dim, depth=self.depth, heads=self.heads, act=act,
            mlp_ratio=self.mlp_ratio, drop_rate=self.dropout
        )
        # Calculate dimensions for node features aggregation
        self.node_features = vertexes * dim
        self.edge_features = vertexes * vertexes * dim
        # MLP to predict a scalar value from aggregated node features
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_features, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, 16), act,
            nn.Linear(16, 1)
        )

    def forward(self, z_e, z_n):
        """
        Forward pass of the Discriminator.
        
        Args:
            z_e (torch.Tensor): Edge features tensor of shape (batch, vertexes, vertexes, edges).
            z_n (torch.Tensor): Node features tensor of shape (batch, vertexes, nodes).
        
        Returns:
            torch.Tensor: Prediction scores (typically a scalar per sample).
        """
        b, n, c = z_n.shape
        # Unpack the shape of edge features (not used further directly)
        _, _, _, d = z_e.shape

        # Process node and edge features separately
        node = self.node_layers(z_n)
        edge = self.edge_layers(z_e)
        # Symmetrize edge features by averaging with its transpose
        edge = (edge + edge.permute(0, 2, 1, 3)) / 2

        # Process features through the Transformer Encoder
        node, edge = self.TransformerEncoder(node, edge)

        # Flatten node features for MLP
        node = node.view(b, -1)
        # Predict a scalar score using the node MLP
        prediction = self.node_mlp(node)

        return prediction


class simple_disc(nn.Module):
    """
    A simplified discriminator that processes flattened features through an MLP
    to predict a scalar score.

    This class is used in NoTarget model.
    """
    def __init__(self, act, m_dim, vertexes, b_dim):
        """
        Initializes the simple discriminator.

        Args:
            act (str): Activation function type ("relu", "leaky", "sigmoid", or "tanh").
            m_dim (int): Dimensionality for atom type features.
            vertexes (int): Number of vertexes.
            b_dim (int): Dimensionality for bond type features.
        """
        super().__init__()

        # Set the activation function and check if it's supported
        if act == "relu":
            act = nn.ReLU()
        elif act == "leaky":
            act = nn.LeakyReLU()
        elif act == "sigmoid":
            act = nn.Sigmoid()
        elif act == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function: {}".format(act))

        # Compute total number of features combining both dimensions
        features = vertexes * m_dim + vertexes * vertexes * b_dim
        print(vertexes)
        print(m_dim)
        print(b_dim)
        print(features)
        self.predictor = nn.Sequential(
            nn.Linear(features, 256), act,
            nn.Linear(256, 128), act,
            nn.Linear(128, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, 16), act,
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        Forward pass of the simple discriminator.
        
        Args:
            x (torch.Tensor): Input features tensor.
        
        Returns:
            torch.Tensor: Prediction scores.
        """
        prediction = self.predictor(x)
        return prediction