import torch
import torch.nn as nn

from src.classifier.gnn.layers import STBlock


# Skeleton based HOI classifier.
# Two ST blocks then mean pool over time and nodes then a small MLP head.
# Kept deliberately small. We have ~1900 segments. A bigger net overfits.
class STGATv2(nn.Module):
    def __init__(self, in_channels=6, num_classes=4, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.block1 = STBlock(in_channels, hidden, heads, dropout=dropout)
        self.block2 = STBlock(hidden, hidden, heads, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, adj):
        # x: (B, T, N, C)
        h = self.block1(x, adj)
        h = self.block2(h, adj)
        h = h.mean(dim=(1, 2))   # global pool over time and nodes -> (B, hidden)
        return self.head(h)
