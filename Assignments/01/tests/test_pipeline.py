import torch
from src.dataset import get_batch, split_ids
from src.generate import generate_text, load_model
from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer


def test_tokenizer_roundtrip_and_determinism():
    text = "hello world\nHELLO"
    a, b = CharTokenizer.from_text(text), CharTokenizer.from_text(text)
    assert a.chars == b.chars == sorted(set(text))
    assert a.decode(a.encode(text)) == text
    assert a.encode("hello zzz") == a.encode("hello ")      # unknown chars are dropped (documented policy)


def test_targets_are_shifted_by_one():
    data = torch.arange(1000)
    x, y = get_batch(data, 4, 16)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])                 # target at t == input at t+1
    assert torch.equal(y, x + 1)


def test_split_is_disjoint():
    tr, va = split_ids(torch.arange(100), 0.1)
    assert len(tr) == 90 and len(va) == 10 and tr.max() < va.min()


def test_checkpoint_reload_and_generate(tmp_path):
    cfg = GPTConfig(vocab_size=12, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    m = MiniGPT(cfg)
    tok = CharTokenizer(list("abcdefghijkl"))
    p = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "config": cfg.to_dict(), "chars": tok.chars}, p)
    m2, tok2 = load_model(str(p))
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    t1 = generate_text(m2, tok2, "abc", 20, temperature=0.0)
    t2 = generate_text(m2, tok2, "abc", 20, temperature=0.0)
    assert t1 == t2 and len(t1) == 23                       # greedy decoding is deterministic


def test_generation_longer_than_context_does_not_crash():
    cfg = GPTConfig(vocab_size=10, context_length=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    m = MiniGPT(cfg)
    out = m.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=30)
    assert out.shape == (1, 33)


def test_learned_position_variant():
    cfg = GPTConfig(vocab_size=10, context_length=8, d_model=16, n_heads=2, n_layers=1, pos_type="learned")
    logits, _ = MiniGPT(cfg)(torch.randint(0, 10, (2, 8)))
    assert logits.shape == (2, 8, 10)
