"""BERT-style masked LM (bidirectional) vs GPT-style causal LM (Milestone 3 / Lesson 06).

Three tiny models with identical blocks (d_model=64, 2 layers, context 64):
  A  causal attention  + next-token objective      (GPT)
  B  bidirectional attn + next-token objective     (what happens if you REMOVE the causal mask)
  C  bidirectional attn + masked-token objective   (BERT-style MLM, 15% [MASK])
The key measurement is the "honest" next-token loss: predict token k given ONLY tokens < k,
which is exactly the situation at generation time.

    python -m experiments.mlm_vs_clm
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.attention_visualization import _save, heat
from src.attention import causal_mask
from src.dataset import get_batch, load_text, split_ids
from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import set_seed

QUICK = bool(os.environ.get("MINIGPT_QUICK"))
CTX, STEPS, BS, LR, LOG_DIR = 64, (60 if QUICK else 400), 32, 2e-3, "outputs/logs"


def make(vocab, causal):
    return MiniGPT(GPTConfig(vocab_size=vocab, context_length=CTX, d_model=64, n_heads=4, n_layers=2,
                             dropout=0.0, causal=causal))


def train_next_token(model, data, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = []
    for s in range(STEPS):
        x, y = get_batch(data, BS, CTX)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 20 == 0 or s == STEPS - 1:
            curve.append((s, loss.item()))
    return curve


def train_mlm(model, data, mask_id, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = []
    for s in range(steps):
        x, _ = get_batch(data, BS, CTX)
        m = torch.rand(x.shape) < 0.15
        m[:, 0] = True if not m.any() else m[:, 0]
        inp = x.clone()
        inp[m] = mask_id
        logits, _ = model(inp)
        loss = F.cross_entropy(logits[m], x[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 20 == 0 or s == steps - 1:
            curve.append((s, loss.item()))
    return curve


@torch.no_grad()
def teacher_forced_loss(model, data, n=20):
    model.eval()
    return float(np.mean([model(*get_batch(data, BS, CTX))[1].item() for _ in range(n)]))


@torch.no_grad()
def honest_loss(model, data, trials=300):
    """Loss for predicting token k from ONLY tokens < k (the generation-time situation)."""
    model.eval()
    ix = torch.randint(len(data) - CTX - 1, (trials,))
    ks = torch.randint(8, CTX, (trials,))
    out = []
    for i, k in zip(ix.tolist(), ks.tolist()):
        logits, _ = model(data[i:i + k].unsqueeze(0))
        out.append(F.cross_entropy(logits[:, -1], data[i + k].view(1)).item())
    return float(np.mean(out))


@torch.no_grad()
def mlm_eval(model, data, mask_id, n=256):
    """Masked-token loss at one position with (a) both-side context, (b) only left context."""
    model.eval()
    k = CTX // 2
    ix = torch.randint(len(data) - CTX - 1, (n,))
    x = torch.stack([data[i:i + CTX] for i in ix.tolist()])
    both = x.clone()
    both[:, k] = mask_id
    left = x.clone()
    left[:, k:] = mask_id
    lb = F.cross_entropy(model(both)[0][:, k], x[:, k]).item()
    ll = F.cross_entropy(model(left)[0][:, k], x[:, k]).item()
    return lb, ll


def main():
    set_seed(0)
    text = load_text("data/input.txt")
    tok = CharTokenizer.from_text(text)
    V = tok.vocab_size
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = split_ids(ids, 0.1)
    counts = torch.bincount(ids, minlength=V).float()
    p = counts / counts.sum()
    unigram = float(-(p * torch.log(p + 1e-12)).sum())

    res = {"uniform_guess_loss": math.log(V), "unigram_entropy_loss": unigram, "steps_A_B": STEPS, "steps_C_mlm": 3 * STEPS}
    set_seed(0)
    A = make(V, True)
    cA = train_next_token(A, train, "A")
    set_seed(0)
    B = make(V, False)
    cB = train_next_token(B, train, "B")
    set_seed(0)
    C = make(V + 1, False)
    cC = train_mlm(C, train, V, 3 * STEPS)   # only ~15% of positions give signal per step, so 3x steps

    res["A_causal_next_token"] = {"teacher_forced_val_loss": teacher_forced_loss(A, val), "honest_val_loss": honest_loss(A, val)}
    res["B_bidirectional_next_token"] = {"teacher_forced_val_loss": teacher_forced_loss(B, val), "honest_val_loss": honest_loss(B, val)}
    lb, ll = mlm_eval(C, val, V)
    res["C_bert_mlm"] = {"masked_loss_both_sides": lb, "masked_loss_left_context_only": ll}
    json.dump(res, open(f"{LOG_DIR}/mlm_vs_clm.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    # ---------- Figure: mask matrices + objectives + measurements ----------
    T = 8
    lab = [str(i + 1) for i in range(T)]
    fig = plt.figure(figsize=(17, 9))
    a1, a2 = fig.add_subplot(2, 3, 1), fig.add_subplot(2, 3, 2)
    heat(a1, np.ones((T, T)), lab, lab, "BERT / MLM mask: every token sees every token", cmap="Greens", fmt="{:.0f}", vmin=0, vmax=1.4)
    heat(a2, causal_mask(T).numpy().astype(float), lab, lab, "GPT / CLM mask: token t sees only tokens <= t", cmap="Greens", fmt="{:.0f}", vmin=0, vmax=1.4)
    for a in (a1, a2):
        a.set_xlabel("key position")
        a.set_ylabel("query position")
    a3 = fig.add_subplot(2, 3, 3)
    a3.axis("off")
    w = text[:48].replace("\n", " ")
    masked = "".join("_" if i in (3, 11, 20, 33, 40) else c for i, c in enumerate(w))
    a3.text(0, 1, "Objectives on the same text\n\nBERT (MLM): hide ~15% of tokens, predict them\n"
            f"  input : {masked}\n  target: only the hidden positions\n  -> uses left AND right context\n\n"
            "GPT (CLM): predict the next token everywhere\n"
            f"  input : {w[:-1]}\n  target: {w[1:]}\n  -> uses ONLY left context", family="monospace", fontsize=7.5, va="top")
    a4 = fig.add_subplot(2, 3, 4)
    for name, c in (("A causal next-token", cA), ("B bidirectional next-token (leaks)", cB), ("C BERT-style MLM (masked positions, 3x steps)", cC)):
        a4.plot([s for s, _ in c], [l for _, l in c], label=name)
    a4.axhline(math.log(V), color="gray", ls=":", label="uniform guess")
    a4.set_title("Training loss: B 'wins' only because it can see the answer")
    a4.set_xlabel("step")
    a4.set_ylabel("loss")
    a4.legend(fontsize=7)
    a4.grid(alpha=.3)
    a5 = fig.add_subplot(2, 3, 5)
    names = ["A causal\nteacher-forced", "A causal\nhonest", "B bidir\nteacher-forced", "B bidir\nhonest"]
    vals = [res["A_causal_next_token"]["teacher_forced_val_loss"], res["A_causal_next_token"]["honest_val_loss"],
            res["B_bidirectional_next_token"]["teacher_forced_val_loss"], res["B_bidirectional_next_token"]["honest_val_loss"]]
    bars = a5.bar(names, vals, color=["#3b82f6", "#3b82f6", "#ef4444", "#ef4444"])
    for b, v in zip(bars, vals):
        a5.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    a5.set_title("Next-token loss: normal evaluation vs honest (prefix-only)")
    a5.set_ylabel("validation loss")
    a6 = fig.add_subplot(2, 3, 6)
    b2 = a6.bar(["MLM: left+right\ncontext", "MLM: left context\nonly"], [lb, ll], color=["#10b981", "#f59e0b"])
    for b, v in zip(b2, [lb, ll]):
        a6.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    a6.set_title("BERT-style model relies on right context")
    a6.set_ylabel("masked-token loss")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "fig_mlm_vs_clm.png")


if __name__ == "__main__":
    main()
