"""Quantitative + qualitative evaluation of a checkpoint.

Produces: validation loss / perplexity / bits-per-char, >=5 fixed-prompt samples, a causal-correctness
check, a generation-strategy comparison, and CLI/API call evidence (Figures 8 and 10).

    python -m experiments.evaluate --checkpoint checkpoints/baseline.pt
"""
import argparse
import json
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.attention_visualization import text_figure
from src.dataset import load_text, split_ids
from src.generate import generate_text, load_model
from src.utils import set_seed

PROMPTS = ["ROMEO:", "To be, or not to be", "KING HENRY:\nWhat", "The future of artificial intelligence",
           "First Citizen:\nBefore we proceed", "JULIET:\nO Romeo"]
LOG_DIR, SAMPLE_DIR = "outputs/logs", "outputs/samples"


@torch.no_grad()
def full_val_loss(model, val_ids, ctx, bs=64):
    """Mean next-token cross-entropy over the WHOLE validation split using non-overlapping windows."""
    model.eval()
    n = (len(val_ids) - 1) // ctx
    x = val_ids[: n * ctx].reshape(n, ctx)
    y = val_ids[1: n * ctx + 1].reshape(n, ctx)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        logits, _ = model(x[i:i + bs])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i:i + bs].reshape(-1), reduction="sum").item()
        cnt += y[i:i + bs].numel()
    return tot / cnt


@torch.no_grad()
def causal_check(model, T=None, trials=5):
    """Max change in logits at positions < k when only tokens >= k are replaced (must be ~0)."""
    model.eval()
    T = T or model.cfg.context_length
    worst = 0.0
    for _ in range(trials):
        a = torch.randint(0, model.cfg.vocab_size, (1, T))
        k = int(torch.randint(1, T, (1,)))
        b = a.clone()
        b[:, k:] = torch.randint(0, model.cfg.vocab_size, (1, T - k))
        la, _ = model(a)
        lb, _ = model(b)
        worst = max(worst, (la[:, :k] - lb[:, :k]).abs().max().item())
    return worst


def distinct_n(text, n):
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / max(1, len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    model, tok = load_model(a.checkpoint)
    ids = torch.tensor(tok.encode(load_text(a.data)), dtype=torch.long)
    _, val = split_ids(ids, 0.1)
    ctx = model.cfg.context_length

    vl = full_val_loss(model, val, ctx)
    res = {"checkpoint": a.checkpoint, "params": model.n_params(), "context_length": ctx,
           "val_loss": vl, "val_perplexity": math.exp(vl), "val_bits_per_char": vl / math.log(2),
           "causal_check_max_logit_change": causal_check(model)}
    print(f"val loss {vl:.4f} | perplexity {math.exp(vl):.2f} | bits/char {vl / math.log(2):.3f} | "
          f"causal check max |dlogit| = {res['causal_check_max_logit_change']:.2e}")

    # --- fixed-prompt samples (Figure 8) ---
    set_seed(1234)
    md = ["# Generated samples (temperature 0.8, top-k 40, fixed prompts, seed 1234)\n"]
    shown = []
    for p in PROMPTS:
        t = generate_text(model, tok, p, 220, 0.8, 40)
        md.append(f"## Prompt: `{p!r}`\n```\n{t}\n```\n")
        shown.append(f"Prompt: {p!r}\n{t}\n")
    open(f"{SAMPLE_DIR}/samples.md", "w").write("\n".join(md))
    text_figure("\n".join(shown[:4]), "fig08_generation_samples.png", "python -m experiments.evaluate  # samples")

    # --- generation-strategy comparison (optional experiment) ---
    strat = []
    for name, temp, k in [("greedy (T=0)", 0.0, None), ("T=0.5", 0.5, 40), ("T=0.8", 0.8, 40), ("T=1.2", 1.2, None)]:
        set_seed(7)
        t = generate_text(model, tok, "ROMEO:", 500, temp, k)
        strat.append({"strategy": name, "distinct_word_1": distinct_n(t, 1), "distinct_word_2": distinct_n(t, 2),
                      "sample": t[:240]})
    res["generation_strategies"] = strat
    with open(f"{SAMPLE_DIR}/strategies.md", "w") as f:
        for s in strat:
            f.write(f"## {s['strategy']}  (distinct-1 {s['distinct_word_1']:.2f}, distinct-2 {s['distinct_word_2']:.2f})\n```\n{s['sample']}\n```\n")
    for s in strat:
        print(f"{s['strategy']:14s} distinct-1 {s['distinct_word_1']:.2f}  distinct-2 {s['distinct_word_2']:.2f}")

    # --- CLI evidence (Figure 10a) ---
    cmd = [sys.executable, "-m", "src.generate", "--checkpoint", a.checkpoint, "--prompt",
           "The future of artificial intelligence", "--max-new-tokens", "80"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    open(f"{LOG_DIR}/cli_call.txt", "w").write(out)
    text_figure(out, "fig10a_cli_call.png", " ".join(["python -m src.generate", "--prompt 'The future of artificial intelligence'", "--max-new-tokens 80"]))

    # --- API evidence (Figure 10b), in-process via FastAPI's TestClient ---
    if not a.no_api:
        try:
            os.environ["CHECKPOINT"] = a.checkpoint
            sys.path.insert(0, os.getcwd())
            from fastapi.testclient import TestClient
            import api
            c = TestClient(api.app)
            r1 = c.get("/health")
            r2 = c.post("/generate", json={"prompt": "ROMEO:", "max_new_tokens": 120, "temperature": 0.8})
            txt = (f"GET  /health   -> {r1.status_code} {r1.json()}\n"
                   f"POST /generate -> {r2.status_code}\n{json.dumps(r2.json(), indent=2)}")
            open(f"{LOG_DIR}/api_call.txt", "w").write(txt)
            text_figure(txt, "fig10b_api_call.png", "curl -X POST localhost:8000/generate -d '{\"prompt\": \"ROMEO:\"}'")
            res["api_ok"] = r2.status_code == 200
        except Exception as e:  # noqa: BLE001
            print("API check skipped:", e)
            res["api_ok"] = None
    json.dump(res, open(f"{LOG_DIR}/evaluation.json", "w"), indent=2)
    print("saved evaluation.json")


if __name__ == "__main__":
    main()
