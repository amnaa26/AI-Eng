import torch
from src.attention import scaled_dot_product_attention, causal_mask, MultiHeadAttention
from src.gpt import GPTConfig, MiniGPT


def test_future_weights_zero():
    q = k = v = torch.randn(1, 6, 8)
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    assert torch.all(torch.triu(w[0], diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 6))


def test_mha_no_future_leak():
    mha = MultiHeadAttention(16, 4).eval()
    x = torch.randn(1, 8, 16)
    y1 = mha(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 16)       # change only future positions
    y2 = mha(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)


def test_gpt_causal():
    m = MiniGPT(GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)).eval()
    a = torch.randint(0, 20, (1, 12))
    b = a.clone()
    b[:, 7:] = torch.randint(0, 20, (1, 5))
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
