import torch.nn as nn

class SingleFrameClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        # Simple MLP with one hidden layer
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, feature_dim)
        return self.net(x)
