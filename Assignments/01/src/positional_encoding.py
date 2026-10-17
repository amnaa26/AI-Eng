"""Sinusoidal positional encoding (Vaswani et al., 2017), implemented from scratch."""
import math
import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...). Shape (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d_model // 2]
    return pe


class LearnedPositionalEncoding(nn.Module):
    """Optional alternative: one trainable vector per position."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        return x + self.emb(torch.arange(x.size(1), device=x.device)).unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[: x.size(1)].unsqueeze(0)
