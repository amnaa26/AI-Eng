import torch
from src.gpt import GPTConfig, MiniGPT


def test_tiny_overfit():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, context_length=16, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    m = MiniGPT(cfg)
    data = torch.randint(0, 20, (4, 17))
    x, y = data[:, :-1], data[:, 1:]         # shifted targets
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = m(x, y)[1].item()
    for _ in range(150):
        loss = m(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first
