import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module consisting of two linear layers with a ReLU activation in between,
    followed by a dropout on the output.

    Attributes:
        fc1 (nn.Linear): The first fully-connected layer.
        act (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): The second fully-connected layer.
        droprateout (nn.Dropout): Dropout layer applied to the output.
    """
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
        """
        Initializes the MLP module.

        Args:
            in_feat (int): Number of input features.
            hid_feat (int, optional): Number of hidden features. Defaults to in_feat if not provided.
            out_feat (int, optional): Number of output features. Defaults to in_feat if not provided.
            dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()

        # Set hidden and output dimensions to input dimension if not specified
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat

        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear layers, activation, and dropout.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)

class MHA(nn.Module):
    """
    Multi-Head Attention (MHA) module of the graph transformer with edge features incorporated into the attention computation.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor for the attention scores.
        q, k, v (nn.Linear): Linear layers to project the node features into query, key, and value embeddings.
        e (nn.Linear): Linear layer to project the edge features.
        d_k (int): Dimension of each attention head.
        out_e (nn.Linear): Linear layer applied to the computed edge features.
        out_n (nn.Linear): Linear layer applied to the aggregated node features.
    """
    def __init__(self, dim, heads, attention_dropout=0.):
        """
        Initializes the Multi-Head Attention module.

        Args:
            dim (int): Dimensionality of the input features.
            heads (int): Number of attention heads.
            attention_dropout (float, optional): Dropout rate for attention (not used explicitly in this implementation).
        """
        super().__init__()

        # Ensure that dimension is divisible by the number of heads
        assert dim % heads == 0

        self.heads = heads
        self.scale = 1. / math.sqrt(dim)  # Scaling factor for attention
        # Linear layers for projecting node features
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # Linear layer for projecting edge features
        self.e = nn.Linear(dim, dim)
        self.d_k = dim // heads  # Dimension per head

        # Linear layers for output transformations
        self.out_e = nn.Linear(dim, dim)
        self.out_n = nn.Linear(dim, dim)

    def forward(self, node, edge):
        """
        Forward pass for the Multi-Head Attention.

        Args:
            node (torch.Tensor): Node feature tensor of shape (batch, num_nodes, dim).
            edge (torch.Tensor): Edge feature tensor of shape (batch, num_nodes, num_nodes, dim).

        Returns:
            tuple: (updated node features, updated edge features)
        """
        b, n, c = node.shape

        # Compute query, key, and value embeddings and reshape for multi-head attention
        q_embed = self.q(node).view(b, n, self.heads, c // self.heads)
        k_embed = self.k(node).view(b, n, self.heads, c // self.heads)
        v_embed = self.v(node).view(b, n, self.heads, c // self.heads)

        # Compute edge embeddings
        e_embed = self.e(edge).view(b, n, n, self.heads, c // self.heads)

        # Adjust dimensions for broadcasting: add singleton dimensions to queries and keys
        q_embed = q_embed.unsqueeze(2)  # Shape: (b, n, 1, heads, c//heads)
        k_embed = k_embed.unsqueeze(1)  # Shape: (b, 1, n, heads, c//heads)

        # Compute  attention scores
        attn = q_embed * k_embed
        attn = attn / math.sqrt(self.d_k)
        attn = attn * (e_embed + 1) * e_embed   # Modulated attention incorporating edge features

        edge_out = self.out_e(attn.flatten(3))  # Flatten last dimension for linear layer

        # Apply softmax over the node dimension to obtain normalized attention weights
        attn = F.softmax(attn, dim=2)

        v_embed = v_embed.unsqueeze(1)  # Adjust dimensions to broadcast: (b, 1, n, heads, c//heads)
        v_embed = attn * v_embed
        v_embed = v_embed.sum(dim=2).flatten(2)
        node_out = self.out_n(v_embed)

        return node_out, edge_out

class Encoder_Block(nn.Module):
    """
    Transformer encoder block that integrates node and edge features.
    
    Consists of:
        - A multi-head attention layer with edge modulation.
        - Two MLP layers, each with residual connections and layer normalization.
    
    Attributes:
        ln1, ln3, ln4, ln5, ln6 (nn.LayerNorm): Layer normalization modules.
        attn (MHA): Multi-head attention module.
        mlp, mlp2 (MLP): MLP modules for further transformation of node and edge features.
    """
    def __init__(self, dim, heads, act, mlp_ratio=4, drop_rate=0.):
        """
        Initializes the encoder block.

        Args:
            dim (int): Dimensionality of the input features.
            heads (int): Number of attention heads.
            act (callable): Activation function (not explicitly used in this block, but provided for potential extensions).
            mlp_ratio (int, optional): Ratio to determine the hidden layer size in the MLP. Defaults to 4.
            drop_rate (float, optional): Dropout rate applied in the MLPs. Defaults to 0.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, heads, drop_rate)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim, dropout=drop_rate)
        self.mlp2 = MLP(dim, dim * mlp_ratio, dim, dropout=drop_rate)
        self.ln5 = nn.LayerNorm(dim)
        self.ln6 = nn.LayerNorm(dim)

    def forward(self, x, y):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Node feature tensor.
            y (torch.Tensor): Edge feature tensor.

        Returns:
            tuple: (updated node features, updated edge features)
        """
        x1 = self.ln1(x)
        x2, y1 = self.attn(x1, y)
        x2 = x1 + x2
        y2 = y + y1
        x2 = self.ln3(x2)
        y2 = self.ln4(y2)
        x = self.ln5(x2 + self.mlp(x2))
        y = self.ln6(y2 + self.mlp2(y2))
        return x, y

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder composed of a sequence of encoder blocks.

    Attributes:
        Encoder_Blocks (nn.ModuleList): A list of Encoder_Block modules stacked sequentially.
    """
    def __init__(self, dim, depth, heads, act, mlp_ratio=4, drop_rate=0.1):
        """
        Initializes the Transformer Encoder.

        Args:
            dim (int): Dimensionality of the input features.
            depth (int): Number of encoder blocks to stack.
            heads (int): Number of attention heads in each block.
            act (callable): Activation function (passed to encoder blocks for potential use).
            mlp_ratio (int, optional): Ratio for determining the hidden layer size in MLP modules. Defaults to 4.
            drop_rate (float, optional): Dropout rate for the MLPs within each block. Defaults to 0.1.
        """
        super().__init__()

        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, act, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

    def forward(self, x, y):
        """
        Forward pass of the Transformer Encoder.

        Args:
            x (torch.Tensor): Node feature tensor.
            y (torch.Tensor): Edge feature tensor.

        Returns:
            tuple: (final node features, final edge features) after processing through all encoder blocks.
        """
        for block in self.Encoder_Blocks:
            x, y = block(x, y)
        return x, y