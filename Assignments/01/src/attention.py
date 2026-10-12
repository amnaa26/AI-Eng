"""Scaled dot-product self-attention and multi-head attention, from scratch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Boolean (T, T) mask. True = allowed to attend (lower triangle incl. diagonal)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    q, k: (..., T, d_k)   v: (..., T, d_v)
    mask: broadcastable to (..., T, T); True = keep, False = block.
    Returns (output (..., T, d_v), weights (..., T, T)).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)      # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.last_weights = None  # stored for visualization

    def split_heads(self, x):
        B, T, _ = x.shape
        # (B, T, d_model) -> (B, T, H, d_head) -> (B, H, T, d_head)
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):
        B, H, T, _ = x.shape
        # (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x, trace: bool = False):
        B, T, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        mask = causal_mask(T, x.device) if self.causal else None
        out, w = scaled_dot_product_attention(q, k, v, mask, self.attn_drop)
        self.last_weights = w.detach()
        merged = self.merge_heads(out)
        result = self.out_proj(merged)
        if trace:
            print(f"input            {tuple(x.shape)}")
            print(f"q/k/v split      {tuple(q.shape)}  (B, H, T, d_head)")
            print(f"scores/weights   {tuple(w.shape)}  (B, H, T, T)")
            print(f"per-head output  {tuple(out.shape)}")
            print(f"merged           {tuple(merged.shape)}  (B, T, d_model)")
            print(f"final projection {tuple(result.shape)}")
        return result
