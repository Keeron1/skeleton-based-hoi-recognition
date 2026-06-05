import torch
import torch.nn as nn


# GATv2 (Brody et al. 2022). Equation in plain language:
#   e_ij = a^T * LeakyReLU(W_l h_i + W_r h_j)
#   alpha_ij = softmax_j(e_ij) over the neighbours of i
#   h_i' = sum_j alpha_ij * W_r h_j
#
# The graph is fixed (18 nodes, ~30 edges) so we run dense attention over
# all node pairs and mask non-edges to -inf before the softmax.
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        self.W_l = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.a)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, F_in)
        # adj: (N, N) bool
        B, N, _ = x.shape
        H, D = self.heads, self.out_dim

        l = self.W_l(x).view(B, N, H, D)
        r = self.W_r(x).view(B, N, H, D)

        # Pairwise: l[i] + r[j] for every (i, j)
        e = l.unsqueeze(2) + r.unsqueeze(1)             # (B, N, N, H, D)
        e = self.leaky(e)
        e = (e * self.a).sum(-1)                         # (B, N, N, H)

        e = e.masked_fill(~adj.view(1, N, N, 1), float("-inf"))
        attn = torch.softmax(e, dim=2)
        attn = self.dropout(attn)

        # Aggregate neighbour values
        out = torch.einsum("bijh,bjhd->bihd", attn, r)   # (B, N, H, D)
        return out.reshape(B, N, H * D)


# Conv along the time axis for each node independently.
# Pads so output length matches input length.
class TemporalConv(nn.Module):
    def __init__(self, channels, kernel=9):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: (B, T, N, C) -> (B, C, T, N) for Conv2d
        x = x.permute(0, 3, 1, 2)
        x = self.bn(self.conv(x))
        return x.permute(0, 2, 3, 1)


# One ST block: spatial GATv2 per frame, then temporal conv across frames.
# Residual when input and output dims match.
class STBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout=0.0):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        head_dim = out_dim // heads

        self.spatial = GATv2Layer(in_dim, head_dim, heads, dropout=dropout)
        self.temporal = TemporalConv(out_dim)
        self.relu = nn.ReLU()
        self.residual = (in_dim == out_dim)

    def forward(self, x, adj):
        # x: (B, T, N, C)
        B, T, N, C = x.shape

        h = x.reshape(B * T, N, C)
        h = self.spatial(h, adj)
        h = h.reshape(B, T, N, -1)
        h = self.relu(h)

        h = self.temporal(h)

        if self.residual:
            h = h + x

        return self.relu(h)
