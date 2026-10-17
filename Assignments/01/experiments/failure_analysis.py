"""Failure analysis (report section 8): measurable limitations of the trained model.

  1. Position-dependent loss  - early characters have little context, so they are predicted worse.
  2. Invented words           - a character-level model can spell non-words; measured per decoding strategy.
  3. Degeneration under greedy decoding - repeated phrases / loops.
  4. Out-of-domain prompts    - the model only knows Shakespeare-style text.

    python -m experiments.failure_analysis --checkpoint checkpoints/baseline.pt
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.attention_visualization import _save
from src.dataset import load_text, split_ids
from src.generate import generate_text, load_model
from src.utils import set_seed

LOG_DIR, SAMPLE_DIR = "outputs/logs", "outputs/samples"
OOD_PROMPTS = ["def main():", "The future of artificial intelligence", "2 + 2 =", "Dear Sir or Madam,"]


@torch.no_grad()
def loss_by_position(model, val_ids, ctx, bs=64):
    model.eval()
    n = (len(val_ids) - 1) // ctx
    x = val_ids[: n * ctx].reshape(n, ctx)
    y = val_ids[1: n * ctx + 1].reshape(n, ctx)
    tot = torch.zeros(ctx)
    for i in range(0, n, bs):
        logits, _ = model(x[i:i + bs])
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i:i + bs].reshape(-1), reduction="none")
        tot += l.view(-1, ctx).sum(0)
    return (tot / n).numpy()


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def repeat_rate(t, n=4):
    w = words(t)
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / max(1, len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    a = ap.parse_args()
    model, tok = load_model(a.checkpoint)
    text = load_text("data/input.txt")
    vocab = set(words(text))
    _, val = split_ids(torch.tensor(tok.encode(text), dtype=torch.long), 0.1)
    ctx = model.cfg.context_length

    pos = loss_by_position(model, val, ctx)
    res = {"loss_first_8_positions": float(pos[:8].mean()), "loss_positions_8_to_32": float(pos[8:32].mean()),
           "loss_last_half": float(pos[ctx // 2:].mean())}

    strat = {}
    for name, temp, k in [("greedy", 0.0, None), ("T=0.8", 0.8, 40), ("T=1.2", 1.2, None)]:
        invented, rep = [], []
        for seed in range(5):
            set_seed(seed)
            t = generate_text(model, tok, "ROMEO:", 600, temp, k)
            w = words(t)
            invented.append(np.mean([x not in vocab for x in w]))
            rep.append(repeat_rate(t))
        strat[name] = {"invented_word_rate": float(np.mean(invented)), "repeated_4gram_rate": float(np.mean(rep))}
    res["strategies"] = strat
    set_seed(0)
    greedy_text = generate_text(model, tok, "ROMEO:", 400, 0.0)
    ood = {p: generate_text(model, tok, p, 160, 0.8, 40) for p in OOD_PROMPTS}
    res["ood_invented_word_rate"] = {p: float(np.mean([x not in vocab for x in words(t)] or [0])) for p, t in ood.items()}
    json.dump(res, open(f"{LOG_DIR}/failure_analysis.json", "w"), indent=2)
    with open(f"{SAMPLE_DIR}/failure_samples.md", "w") as f:
        f.write("## Greedy decoding from 'ROMEO:'\n```\n" + greedy_text + "\n```\n")
        for p, t in ood.items():
            f.write(f"## Out-of-domain prompt {p!r}\n```\n{t}\n```\n")
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    ax[0].plot(pos)
    ax[0].set_xlabel("position inside the context window")
    ax[0].set_ylabel("mean next-token loss")
    ax[0].set_title("Failure 1: little context => higher loss")
    ax[0].grid(alpha=.3)
    names = list(strat)
    xs = np.arange(len(names))
    ax[1].bar(xs - .2, [strat[n]["invented_word_rate"] for n in names], .4, label="invented-word rate")
    ax[1].bar(xs + .2, [strat[n]["repeated_4gram_rate"] for n in names], .4, label="repeated 4-gram rate")
    ax[1].set_xticks(xs, names)
    ax[1].set_title("Failure 2 & 3: non-words and repetition by decoding strategy")
    ax[1].legend()
    ood_r = res["ood_invented_word_rate"]
    ax[2].barh(list(ood_r), list(ood_r.values()), color="#ef4444")
    ax[2].set_title("Failure 4: out-of-domain prompts (invented-word rate)")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "fig_failure_modes.png")


if __name__ == "__main__":
    main()
