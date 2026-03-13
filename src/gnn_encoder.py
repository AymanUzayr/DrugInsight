import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    def __init__(self, 
                 in_channels=8,       # atom feature size
                 edge_dim=6,          # bond feature size
                 hidden_channels=128,
                 out_channels=256,    # embedding size
                 num_layers=4,
                 num_timesteps=2,
                 dropout=0.2):
        super().__init__()

        self.attentive_fp = AttentiveFP(
            in_channels      = in_channels,
            hidden_channels  = hidden_channels,
            out_channels     = out_channels,
            edge_dim         = edge_dim,
            num_layers       = num_layers,
            num_timesteps    = num_timesteps,
            dropout          = dropout,
        )

    def forward(self, data):
        return self.attentive_fp(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )