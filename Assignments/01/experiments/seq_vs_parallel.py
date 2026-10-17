"""Lesson 01: why Transformers? Measured sequential-vs-parallel comparison.

Compares one recurrent layer (GRU cell applied step by step: sequential depth = T) with one self-attention
layer (sequential depth = 1, but a T x T score matrix). Run on an otherwise idle machine.

    python -m experiments.seq_vs_parallel
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from experiments.attention_visualization import _save
from src.attention import MultiHeadAttention

B, D, H = 16, 128, 4
LENGTHS = [32, 64, 128] if os.environ.get("MINIGPT_QUICK") else [32, 64, 128, 256, 512]


def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def main():
    torch.manual_seed(0)
    cell, gru, attn = nn.GRUCell(D, D), nn.GRU(D, D, batch_first=True), MultiHeadAttention(D, H)

    def loop(x):
        h = torch.zeros(x.size(0), D)
        outs = []
        for t in range(x.size(1)):          # step t needs h from step t-1: cannot be parallelised over t
            h = cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, 1)

    rows = []
    for T in LENGTHS:
        x = torch.randn(B, T, D)
        run = lambda f: (lambda: f(x).sum().backward())
        t_loop, t_gru, t_att = timeit(run(loop)), timeit(run(lambda z: gru(z)[0])), timeit(run(attn))
        rows.append({"T": T, "rnn_cell_loop_s": t_loop, "fused_gru_s": t_gru, "attention_s": t_att,
                     "rnn_sequential_steps": T, "attention_sequential_steps": 1,
                     "rnn_state_mb": B * T * D * 4 / 2**20, "attention_scores_mb": B * H * T * T * 4 / 2**20})
        print(rows[-1])
    json.dump(rows, open("outputs/logs/seq_vs_parallel.json", "w"), indent=2)
    md = ["| T | RNN loop fwd+bwd (s) | fused GRU (s) | self-attention (s) | RNN seq. steps | attn seq. steps | RNN hidden states (MB) | attention scores (MB) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['T']} | {r['rnn_cell_loop_s']:.3f} | {r['fused_gru_s']:.3f} | {r['attention_s']:.3f} | {r['T']} | 1 | "
                  f"{r['rnn_state_mb']:.2f} | {r['attention_scores_mb']:.2f} |")
    open("outputs/logs/seq_vs_parallel.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    Ts = [r["T"] for r in rows]
    ax[0].plot(Ts, [r["rnn_cell_loop_s"] for r in rows], "o-", label="RNN (GRU cell loop)")
    ax[0].plot(Ts, [r["fused_gru_s"] for r in rows], "s-", label="fused nn.GRU")
    ax[0].plot(Ts, [r["attention_s"] for r in rows], "^-", label="self-attention")
    ax[0].set_xlabel("sequence length T")
    ax[0].set_ylabel("forward+backward time (s)")
    ax[0].set_title("Measured time (CPU, batch 16, d=128)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(Ts, [r["rnn_state_mb"] for r in rows], "o-", label="RNN hidden states  O(T d)")
    ax[1].plot(Ts, [r["attention_scores_mb"] for r in rows], "^-", label="attention scores  O(H T^2)")
    ax[1].set_xlabel("sequence length T")
    ax[1].set_ylabel("MB (float32)")
    ax[1].set_title("The trade-off: parallel depth costs T^2 memory")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.patch.set_facecolor("white")
    _save(fig, "fig_seq_vs_parallel.png")


if __name__ == "__main__":
    main()
