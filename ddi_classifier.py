import torch
import torch.nn as nn

class DDIClassifier(nn.Module):
    def __init__(self,
                 drug_embed_dim=256,   # GNN output size
                 extra_features=5,     # rule flags + twosides score
                 num_heads=4,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()

        # Input dim = drug_A + drug_B + extra features
        input_dim = drug_embed_dim * 2 + extra_features

        # Project to transformer-friendly dim
        self.input_proj = nn.Linear(input_dim, 256)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = 256,
            nhead      = num_heads,
            dim_feedforward = 512,
            dropout    = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification heads
        self.prob_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
)

        self.severity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)        # Minor / Moderate / Major logits
        )

    def forward(self, embed_a, embed_b, extra):
        # Concatenate drug embeddings + rule/signal features
        x = torch.cat([embed_a, embed_b, extra], dim=-1)

        # Project + add sequence dim for transformer
        x = self.input_proj(x).unsqueeze(1)

        # Transformer
        x = self.transformer(x).squeeze(1)

        # Heads
        prob     = self.prob_head(x)
        severity = self.severity_head(x)

        return prob, severity