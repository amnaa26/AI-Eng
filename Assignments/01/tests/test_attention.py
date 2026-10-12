import torch
from src.attention import scaled_dot_product_attention, causal_mask, MultiHeadAttention


def test_hand_example_matches():
    Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    V = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 2) and w.shape == (3, 3)
    assert torch.allclose(w.sum(-1), torch.ones(3))
    scores = Q @ K.T / (2 ** 0.5)
    assert torch.allclose(w, torch.softmax(scores, -1))


def test_mha_shapes():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    assert mha(x).shape == (2, 10, 32)
    assert mha.last_weights.shape == (2, 4, 10, 10)
