"""MiniGPT: token embedding + sinusoidal positions + causal blocks + LM head."""
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 65
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    tie_weights: bool = False

    def to_dict(self):
        return asdict(self)


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.context_length)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult * cfg.d_model, cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) = idx shifted left by one (next tokens)."""
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)
        x = self.drop(self.pos(x))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))           # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:                       # greedy
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
