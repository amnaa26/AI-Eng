"""Collect ablation results from outputs/logs/*.json -> table (md/csv) + Figures 6 and 9.

    python -m experiments.ablation
"""
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.attention_visualization import _save
from experiments.evaluate import full_val_loss
from src.dataset import load_text, split_ids
from src.generate import load_model

LOG_DIR = "outputs/logs"
GROUPS = {"Depth (n_layers)": [("depth2", "2"), ("baseline", "4"), ("depth6", "6")],
          "Heads (n_heads)": [("heads2", "2"), ("baseline", "4"), ("heads8", "8")],
          "Context length": [("ctx64", "64"), ("baseline", "128"), ("ctx256", "256")],
          "Position method": [("baseline", "sinusoidal"), ("learnedpos", "learned")]}


def load(tag):
    p = f"{LOG_DIR}/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fig_loss_curves(tag="baseline"):
    s = load(tag)
    steps = [r["step"] for r in s["log"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(steps, [r["train_loss"] for r in s["log"]], "o-", label="train loss")
    ax[0].plot(steps, [r["val_loss"] for r in s["log"]], "s-", label="validation loss")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("cross-entropy (nats)")
    ax[0].set_title(f"Baseline: train vs validation loss (final val {s['final_val_loss']:.3f})")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(steps, [r["lr"] for r in s["log"]], "o-", color="#b45309")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("learning rate")
    ax[1].set_title("Learning-rate schedule (linear warm-up + cosine decay)")
    ax[1].grid(alpha=.3)
    fig.patch.set_facecolor("white")
    _save(fig, "fig06_train_val_loss.png")


def main():
    text = load_text("data/input.txt")
    rows = []
    cache = {}
    for group, items in GROUPS.items():
        for tag, setting in items:
            s = load(tag)
            if s is None:
                continue
            if tag not in cache:
                model, tok = load_model(f"checkpoints/{tag}.pt")
                ids = torch.tensor(tok.encode(text), dtype=torch.long)
                _, val = split_ids(ids, 0.1)
                cache[tag] = full_val_loss(model, val, model.cfg.context_length)
            vl = cache[tag]
            tc = s["train_cfg"]
            rows.append({"group": group, "setting": setting, "tag": tag, "params": s["params"],
                         "batch_size": tc["batch_size"], "context": s["model_cfg"]["context_length"],
                         "final_train_loss": s["final_train_loss"], "full_val_loss": vl,
                         "val_perplexity": math.exp(vl), "train_time_s": s["train_time_s"],
                         "tokens_per_s": s["tokens_per_s"], "peak_mem_mb": s["peak_mem_mb"]})
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(f"{LOG_DIR}/ablation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["| Experiment | Setting | Params | Batch x ctx | Final train loss | Val loss | Val ppl | Train time (s) | Tokens/s | Peak RSS (MB) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['group']} | {r['setting']} | {r['params']:,} | {r['batch_size']}x{r['context']} | {r['final_train_loss']:.3f} | "
                  f"{r['full_val_loss']:.3f} | {r['val_perplexity']:.2f} | {r['train_time_s']:.0f} | {r['tokens_per_s']:,.0f} | {r['peak_mem_mb']:.0f} |")
    open(f"{LOG_DIR}/ablation_table.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    if load("baseline"):
        fig_loss_curves("baseline")
    fig, ax = plt.subplots(2, 4, figsize=(19, 8))
    for j, (group, items) in enumerate(GROUPS.items()):
        for tag, setting in items:
            s = load(tag)
            if s:
                lg = s["log"][1:]                      # skip step 0 (untrained, loss ~ ln V) so the curves are readable
                ax[0, j].plot([r["step"] for r in lg], [r["val_loss"] for r in lg], label=f"{setting}")
        ax[0, j].set_title(f"{group}: validation loss during training", fontsize=9)
        ax[0, j].set_xlabel("step")
        ax[0, j].set_ylabel("val loss")
        ax[0, j].grid(alpha=.3)
        ax[0, j].legend()
        rr = [r for r in rows if r["group"] == group]
        bars = ax[1, j].bar([r["setting"] for r in rr], [r["full_val_loss"] for r in rr], color="#3b82f6")
        lo = min(r["full_val_loss"] for r in rr)
        ax[1, j].set_ylim(lo - 0.15, max(r["full_val_loss"] for r in rr) + 0.08)
        for b, r in zip(bars, rr):
            ax[1, j].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['full_val_loss']:.3f}\n{r['params'] / 1e6:.2f}M\n{r['train_time_s']:.0f}s",
                          ha="center", va="bottom", fontsize=8)
        ax[1, j].set_title("final validation loss, full val split\n(y-axis truncated; labels: loss / params / train time)", fontsize=9)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "fig09_ablation.png")


if __name__ == "__main__":
    main()
