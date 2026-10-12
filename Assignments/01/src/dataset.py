"""Next-token prediction data: fixed-length windows with targets shifted by one."""
import os
import urllib.request
import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_text(path="data/input.txt", download=True) -> str:
    if not os.path.exists(path) and download:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_ids(ids, val_frac=0.1):
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device="cpu"):
    """x[t] = data[i+t]; y[t] = data[i+t+1]  ->  target at position t is the NEXT token."""
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)
