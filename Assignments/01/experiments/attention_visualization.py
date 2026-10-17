"""Evidence figures 1-5 and 7 plus the architecture diagram.

Run (after training the baseline):  python -m experiments.attention_visualization
"""
import argparse
import contextlib
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from experiments import hand_calc
from src.attention import MultiHeadAttention, causal_mask
from src.generate import load_model
from src.gpt import GPTConfig, MiniGPT
from src.positional_encoding import sinusoidal_table

FIG_DIR, LOG_DIR = "outputs/figures", "outputs/logs"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", f"{FIG_DIR}/{name}")


def text_figure(text, name, title=None):
    """Render monospace terminal-style text as an image (used as screenshot evidence)."""
    lines = text.rstrip().split("\n")
    w = max(len(l) for l in lines)
    fig, ax = plt.subplots(figsize=(0.075 * w + 0.5, 0.16 * (len(lines) + (2 if title else 0)) + 0.5))
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    if title:
        text = f"$ {title}\n\n" + text
    ax.text(0, 1, text, color="#e2e8f0", family="monospace", fontsize=8, va="top", ha="left")
    _save(fig, name)


def heat(ax, M, xl, yl, title, cmap="Blues", fmt="{:.2f}", vmin=None, vmax=None):
    A = (M.detach().cpu().numpy() if hasattr(M, "detach") else np.asarray(M)).astype(float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e5e7eb")
    ax.imshow(np.where(np.isinf(A), np.nan, A), cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xl)), xl)
    ax.set_yticks(range(len(yl)), yl)
    ax.set_title(title, fontsize=10)
    finite = A[np.isfinite(A)]
    mid = (np.nanmin(finite) + np.nanmax(finite)) / 2 if vmin is None else (vmin + vmax) / 2
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            s = "-inf" if np.isinf(A[i, j]) else fmt.format(A[i, j])
            ax.text(j, i, s, ha="center", va="center", fontsize=9,
                    color="white" if (np.isfinite(A[i, j]) and A[i, j] > mid) else "#111827")


# ---------------- Figure 1: hand calculation ----------------
def fig_hand_calc():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = hand_calc.run(verbose=True)
    text_figure(buf.getvalue(), "fig01a_hand_calc_printout.png", "python -m experiments.hand_calc")
    toks, dims = hand_calc.tokens, ["d1", "d2"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    heat(ax[0, 0], r["S_scaled"], toks, toks, "(2) QK^T / sqrt(d_k)")
    heat(ax[0, 1], r["W"], toks, toks, "(3) softmax weights (rows sum to 1)", vmin=0, vmax=1)
    heat(ax[0, 2], r["out"], dims, toks, "(4) output = weights @ V", cmap="Greens")
    heat(ax[1, 0], r["S_masked"], toks, toks, "(6) causal-masked scores")
    heat(ax[1, 1], r["Wc"], toks, toks, "(6) causal weights", vmin=0, vmax=1)
    heat(ax[1, 2], r["out_c"], dims, toks, "(7) causal output", cmap="Greens")
    fig.suptitle("Hand-calculated attention for 'I love AI' (rows = query tokens, columns = key tokens)")
    fig.patch.set_facecolor("white")
    _save(fig, "fig01b_hand_calc_heatmaps.png")


# ---------------- Figure 2: causal mask ----------------
def fig_causal_mask(T=10):
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 4, causal=True).eval()
    x = torch.randn(1, T, 32)
    mha(x)
    w_causal = mha.last_weights[0, 0].numpy()
    mha.causal = False
    mha(x)
    w_bi = mha.last_weights[0, 0].numpy()
    future = float(np.triu(w_causal, 1).max())
    labels = [str(i + 1) for i in range(T)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    heat(ax[0], causal_mask(T).numpy().astype(float), labels, labels, "Causal mask (1 = may attend)",
         cmap="Greys", fmt="{:.0f}", vmin=0, vmax=1.4)
    ax[1].imshow(w_causal, cmap="viridis")
    ax[1].set_title(f"GPT / causal attention weights\nmax weight on a future token = {future:.1f}", fontsize=10)
    ax[2].imshow(w_bi, cmap="viridis")
    ax[2].set_title("Same weights WITHOUT mask (BERT-style, bidirectional)\nupper triangle = information from the future", fontsize=10)
    for a in ax[1:]:
        a.set_xticks(range(T), labels)
        a.set_yticks(range(T), labels)
        a.set_xlabel("key position")
        a.set_ylabel("query position")
    fig.patch.set_facecolor("white")
    _save(fig, "fig02_causal_mask.png")


# ---------------- Figure 3: multi-head shape trace ----------------
def fig_shape_trace(d_model=128, n_heads=4, B=2, T=10):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model, n_heads).eval()
    x = torch.randn(B, T, d_model)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"B={B} T={T} d_model={d_model} n_heads={n_heads} d_head={d_model // n_heads}")
        y = mha(x, trace=True)
    heads = mha.split_heads(mha.q_proj(x))
    ok_merge = torch.allclose(mha.merge_heads(heads), mha.q_proj(x))
    manual = torch.cat([heads[:, h] for h in range(n_heads)], dim=-1)   # concatenate heads explicitly
    ok_cat = torch.allclose(manual, mha.merge_heads(heads))
    buf.write(f"\nmerge_heads(split_heads(z)) == z            : {ok_merge}\n")
    buf.write(f"merge_heads(...) == concat of the {n_heads} heads: {ok_cat}\n")
    buf.write(f"attention rows sum to 1 in every head          : "
              f"{bool(torch.allclose(mha.last_weights.sum(-1), torch.ones(B, n_heads, T)))}\n")
    buf.write(f"future weights are exactly 0 in every head     : "
              f"{bool((torch.triu(mha.last_weights, 1) == 0).all())}\n")
    os.makedirs(LOG_DIR, exist_ok=True)
    open(f"{LOG_DIR}/shape_trace.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
    text_figure(buf.getvalue(), "fig03_multihead_shape_trace.png", "python -m experiments.attention_visualization  # shape trace")


# ---------------- Figure 4: positional encoding ----------------
def fig_positional_encoding(max_len=64, d_model=128):
    pe = sinusoidal_table(max_len, d_model).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu")
    ax[0].set_title("Sinusoidal PE matrix (position x dimension)")
    ax[0].set_xlabel("dimension")
    ax[0].set_ylabel("position")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for d in (0, 1, 8, 9, 32, 33):
        ax[1].plot(pe[:, d], label=f"dim {d} ({'sin' if d % 2 == 0 else 'cos'})")
    ax[1].set_title("Different dimensions = different wavelengths")
    ax[1].set_xlabel("position")
    ax[1].legend(fontsize=7)
    sim = pe @ pe.T
    sim = sim / np.sqrt(np.outer(np.diag(sim), np.diag(sim)))
    im2 = ax[2].imshow(sim, cmap="viridis")
    ax[2].set_title("Cosine similarity between positions\n(nearby positions are more similar)")
    ax[2].set_xlabel("position")
    ax[2].set_ylabel("position")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    fig.patch.set_facecolor("white")
    _save(fig, "fig04_positional_encoding.png")


# ---------------- Figure 5: model summary ----------------
def model_summary(model):
    c = model.cfg
    n = lambda m: sum(p.numel() for p in m.parameters())
    L = [f"MiniGPT  vocab_size={c.vocab_size}  context_length={c.context_length}  d_model={c.d_model}  "
         f"n_heads={c.n_heads}  n_layers={c.n_layers}  ffn_dim={c.ffn_mult * c.d_model}  dropout={c.dropout}  pos={c.pos_type}",
         "",
         f"{'module':38s}{'detail':44s}{'params':>10s}", "-" * 92,
         f"{'token embedding':38s}{f'({c.vocab_size}, {c.d_model})':44s}{n(model.tok_emb):>10,}",
         f"{'positional encoding':38s}{c.pos_type:44s}{n(model.pos):>10,}"]
    for i, b in enumerate(model.blocks):
        L += [f"{f'block {i}: LayerNorm 1':38s}{f'({c.d_model})':44s}{n(b.ln1):>10,}",
              f"{f'block {i}: multi-head attention':38s}{f'{c.n_heads} heads x d_head {c.d_model // c.n_heads}, Q/K/V/out proj':44s}{n(b.attn):>10,}",
              f"{f'block {i}: LayerNorm 2':38s}{f'({c.d_model})':44s}{n(b.ln2):>10,}",
              f"{f'block {i}: feed-forward (GELU)':38s}{f'{c.d_model} -> {c.ffn_mult * c.d_model} -> {c.d_model}':44s}{n(b.ffn):>10,}"]
    L += [f"{'final LayerNorm':38s}{f'({c.d_model})':44s}{n(model.ln_f):>10,}",
          f"{'LM head':38s}{f'{c.d_model} -> {c.vocab_size} (no bias)':44s}{n(model.lm_head):>10,}",
          "-" * 92, f"{'TOTAL TRAINABLE PARAMETERS':82s}{model.n_params():>10,}",
          f"({model.n_params() / 1e6:.3f}M)"]
    return "\n".join(L)


def fig_model_summary(model):
    s = model_summary(model)
    open(f"{LOG_DIR}/model_summary.txt", "w").write(s + "\n")
    print(s)
    text_figure(s, "fig05_model_summary.png")


# ---------------- Architecture diagram ----------------
def fig_architecture(cfg):
    d, f, h = cfg.d_model, cfg.ffn_mult * cfg.d_model, cfg.n_heads
    # (key, label, colour, tensor shape, box height), listed bottom -> top
    spec = [("ids", "input token ids", "#e2e8f0", "(B, T)", 1.0),
            ("emb", f"token embedding   {cfg.vocab_size} -> {d},   x sqrt(d_model)", "#dbeafe", "(B, T, d)", 1.0),
            ("pos", f"+ {cfg.pos_type} positional encoding", "#dbeafe", "(B, T, d)", 1.0),
            ("drop", f"dropout (p = {cfg.dropout})", "#f1f5f9", "(B, T, d)", 1.0),
            ("ln1", "LayerNorm", "#fef9c3", "(B, T, d)", 1.0),
            ("mha", f"causal multi-head self-attention\n{h} heads, d_head = {d // h}\nsoftmax(QK^T / sqrt(d_k) + mask) V -> concat -> W_O",
             "#fecaca", "(B, T, d)", 1.9),
            ("add1", "+ residual", "#e2e8f0", "(B, T, d)", 1.0),
            ("ln2", "LayerNorm", "#fef9c3", "(B, T, d)", 1.0),
            ("ffn", f"feed-forward\nLinear {d} -> {f}  -  GELU  -  Linear {f} -> {d}", "#bbf7d0", "(B, T, d)", 1.5),
            ("add2", "+ residual", "#e2e8f0", "(B, T, d)", 1.0),
            ("lnf", "final LayerNorm", "#fef9c3", "(B, T, d)", 1.0),
            ("head", f"LM head   Linear {d} -> {cfg.vocab_size}", "#ddd6fe", "(B, T, V)", 1.0),
            ("loss", "softmax -> next-token distribution\nloss = cross-entropy(logits[t], token[t+1])", "#e2e8f0", "", 1.5)]
    gap, block_pad = 0.8, 0.45
    pos, y = {}, 0.5
    for key, _, _, _, ht in spec:
        if key in ("ln1", "lnf"):
            y += block_pad                      # extra room for the dashed block outline
        pos[key] = (y, y + ht)
        y += ht + gap
    total = y
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * total + 1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, total + 0.8)
    ax.axis("off")
    X0, W = 2.6, 7.0
    for key, text, color, shape, ht in spec:
        y0, y1 = pos[key]
        ax.add_patch(FancyBboxPatch((X0, y0), W, ht, boxstyle="round,pad=0.04,rounding_size=0.18", fc=color, ec="#334155", lw=1.2))
        ax.text(X0 + W / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=8.5)
        if shape:
            ax.text(0.1, (y0 + y1) / 2, shape, ha="left", va="center", fontsize=8, color="#475569", family="monospace")
    keys = [k for k, *_ in spec]
    for a, b in zip(keys[:-1], keys[1:]):
        ax.annotate("", xy=(X0 + W / 2, pos[b][0] - 0.02), xytext=(X0 + W / 2, pos[a][1] + 0.02),
                    arrowprops=dict(arrowstyle="->", color="#334155"))

    def skip(src_gap_key, dst_key):          # residual path: leaves the stream before a sub-layer, re-joins after it
        ya = pos[src_gap_key][1] + gap / 2
        yb = (pos[dst_key][0] + pos[dst_key][1]) / 2
        xr = X0 + W + 0.9
        ax.plot([X0 + W / 2 + 0.0, xr, xr], [ya, ya, yb], color="#b45309", lw=1.4, ls="-")
        ax.annotate("", xy=(X0 + W, yb), xytext=(xr, yb), arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.4))
        ax.text(xr + 0.15, (ya + yb) / 2, "skip", rotation=90, va="center", fontsize=7.5, color="#b45309")
    skip("drop", "add1")
    skip("add1", "add2")
    bx0, bx1 = X0 - 0.5, X0 + W + 0.5
    by0, by1 = pos["ln1"][0] - 0.35, pos["add2"][1] + 0.35
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0 + 1.2, by1 - by0, fill=False, ls="--", ec="#64748b"))
    ax.text(bx0 + 0.15, by1 + 0.12, f"Transformer block  x {cfg.n_layers}", fontsize=8.5, color="#334155", va="bottom")
    ax.set_title("MiniGPT architecture (pre-LayerNorm, decoder-only)", fontsize=11)
    fig.patch.set_facecolor("white")
    _save(fig, "fig_architecture.png")


# ---------------- Figure 7: attention of the trained model ----------------
@torch.no_grad()
def fig_trained_attention(model, tok, prompt, layer=0, head=0):
    model.eval()
    ids = torch.tensor([tok.encode(prompt)])
    model(ids)
    chars = [{"\n": "\\n", " ": "_"}.get(c, c) for c in prompt]
    T = len(chars)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    fig, axes = plt.subplots(L, H, figsize=(3.3 * H, 3.2 * L))
    stats = {}
    for l in range(L):
        W = model.blocks[l].attn.last_weights[0].numpy()          # (H, T, T)
        for h in range(H):
            a = axes[l, h]
            a.imshow(W[h], cmap="viridis", vmin=0)
            a.set_title(f"layer {l}  head {h}", fontsize=8)
            a.set_xticks(range(T), chars, fontsize=3.5, rotation=90)
            a.set_yticks(range(T), chars, fontsize=3.5)
            w = W[h]
            idx = np.arange(1, T)
            stats[f"L{l}H{h}"] = {
                "self": float(np.mean(np.diag(w))),
                "previous_token": float(np.mean(w[idx, idx - 1])),
                "first_token": float(np.mean(w[1:, 0])),
                "mean_distance": float(np.mean([(w[t, :t + 1] * (t - np.arange(t + 1))).sum() for t in range(T)]))}
    fig.suptitle(f"Trained-model attention (rows = query, columns = key). Prompt: {prompt!r}", fontsize=10)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, "fig07_trained_attention_grid.png")
    W = model.blocks[layer].attn.last_weights[0, head].numpy()
    fig, a = plt.subplots(figsize=(8.5, 7.5))
    a.imshow(W, cmap="viridis", vmin=0)
    a.set_xticks(range(T), chars, fontsize=7)
    a.set_yticks(range(T), chars, fontsize=7)
    a.set_xlabel("key (attended-to) character")
    a.set_ylabel("query character")
    a.set_title(f"Layer {layer}, head {head}: lower-triangular, so no future tokens are used")
    fig.patch.set_facecolor("white")
    _save(fig, "fig07b_trained_attention_head.png")
    os.makedirs(LOG_DIR, exist_ok=True)
    json.dump(stats, open(f"{LOG_DIR}/attention_stats.json", "w"), indent=2)
    print(f"{'head':7s}{'self':>8s}{'prev':>8s}{'first':>8s}{'avg dist':>10s}")
    for k, v in stats.items():
        print(f"{k:7s}{v['self']:8.3f}{v['previous_token']:8.3f}{v['first_token']:8.3f}{v['mean_distance']:10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    ap.add_argument("--prompt", default="ROMEO:\nWhat light through yonder window breaks")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    fig_hand_calc()
    fig_causal_mask()
    fig_shape_trace()
    fig_positional_encoding()
    if os.path.exists(a.checkpoint):
        model, tok = load_model(a.checkpoint)
        fig_model_summary(model)
        fig_architecture(model.cfg)
        fig_trained_attention(model, tok, a.prompt, a.layer, a.head)
    else:
        print("checkpoint not found; skipped figures 5, 7 and the architecture diagram")


if __name__ == "__main__":
    main()
