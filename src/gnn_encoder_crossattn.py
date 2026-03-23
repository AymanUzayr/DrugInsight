import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from torch_geometric.data import Data, Batch


class GNNEncoderCrossAttn(nn.Module):
    """
    GATv2Conv-based molecular encoder that returns atom-level embeddings.
    Used with cross-attention classifier — does NOT pool to a single vector.

    Input:  PyG Data/Batch with x [N, 8], edge_index, edge_attr [E, 6]
    Output: atom embeddings [N, out_channels]
    """

    def __init__(self,
                 in_channels=8,        # atom feature size
                 edge_dim=6,           # bond feature size
                 hidden_channels=128,
                 out_channels=256,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.2):
        super().__init__()

        self.dropout    = dropout
        self.num_layers = num_layers

        # Input projection — map raw atom features to hidden dim
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GATv2Conv layers with edge features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = hidden_channels
            out_ch = hidden_channels // num_heads  # per-head dim
            self.convs.append(
                GATv2Conv(
                    in_channels  = in_ch,
                    out_channels = out_ch,
                    heads        = num_heads,
                    edge_dim     = edge_dim,
                    dropout      = dropout,
                    concat       = True,       # concat heads → hidden_channels
                    add_self_loops = True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Final projection to output dim
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.act         = nn.ELU()
        self.drop        = nn.Dropout(dropout)

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr

        # Project input
        x = self.act(self.input_proj(x))

        # GATv2 layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)   # residual connection
            x = self.act(x)
            x = self.drop(x)

        # Project to output dim
        x = self.output_proj(x)   # [N_total, out_channels]

        return x  # atom-level embeddings, NOT pooled
