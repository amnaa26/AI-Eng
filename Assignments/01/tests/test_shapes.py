import torch
from src.gpt import GPTConfig, MiniGPT
from src.positional_encoding import sinusoidal_table
from src.transformer_block import TransformerBlock


def test_block_shape():
    assert TransformerBlock(32, 4, 128)(torch.randn(2, 9, 32)).shape == (2, 9, 32)


def test_gpt_logits_and_loss():
    cfg = GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2)
    m = MiniGPT(cfg)
    x = torch.randint(0, 30, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 30) and loss.item() > 0


def test_positional_encoding():
    pe = sinusoidal_table(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
