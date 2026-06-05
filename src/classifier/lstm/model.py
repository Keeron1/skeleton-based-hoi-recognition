import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 num_classes,
                 num_layers=2,
                 dropout=0.3,
                 bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # input shape: (batch, seq_len, feature_dim)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Output of LSTM is doubled when bidirectional
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Dropout before classification head
        self.dropout = nn.Dropout(dropout)

        # Fully connected classification head
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)

        # Take the last hidden state from the final layer
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits
