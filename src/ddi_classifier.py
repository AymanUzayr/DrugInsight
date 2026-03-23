import torch
import torch.nn as nn


class DDIClassifier(nn.Module):
    def __init__(self,
                 drug_embed_dim=256,
                 extra_features=6,
                 dropout=0.5):
        super().__init__()

        input_dim = drug_embed_dim * 2 + extra_features  # 518
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Interaction probability head
        self.prob_head = nn.Linear(128, 1)

        # Severity head — Minor / Moderate / Major
        # NOTE: not trained until severity labels are available
        self.severity_head = nn.Linear(128, 3)

    def forward(self, embed_a, embed_b, extra):
        x = torch.cat([embed_a, embed_b, extra], dim=-1)
        x = self.trunk(x)

        prob     = self.prob_head(x)
        severity = self.severity_head(x)

        return prob, severity