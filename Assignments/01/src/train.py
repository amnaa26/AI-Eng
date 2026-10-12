"""Train MiniGPT. Usage:
    python -m src.train --config configs/tiny_gpt.yaml [--overrides n_layers=2 ...] [--tag depth2]
"""
import argparse
import json
import math
import os
import time
import torch
from .tokenizer import CharTokenizer
from .dataset import load_text, split_ids, get_batch
from .gpt import GPTConfig, MiniGPT
from .utils import set_seed, get_device, load_config


def lr_at(step, tc):
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, tc["max_steps"] - tc["warmup_steps"])
    return 0.1 * tc["lr"] + 0.5 * 0.9 * tc["lr"] * (1 + math.cos(math.pi * p))


@torch.no_grad()
def estimate_loss(model, data, tc, ctx, device):
    model.eval()
    out = {}
    for name, d in data.items():
        losses = torch.zeros(tc["eval_iters"])
        for i in range(tc["eval_iters"]):
            x, y = get_batch(d, tc["batch_size"], ctx, device)
            losses[i] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def apply_overrides(cfg, overrides):
    for o in overrides or []:
        k, v = o.split("=")
        for section in ("model", "train"):
            if k in cfg[section]:
                cfg[section][k] = type(cfg[section][k])(v) if not isinstance(cfg[section][k], bool) else v == "true"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_gpt.yaml")
    ap.add_argument("--overrides", nargs="*")
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a.overrides)
    mc, tc = cfg["model"], cfg["train"]
    set_seed(tc["seed"])
    device = get_device()

    text = load_text(cfg["data"]["path"])
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_d, val_d = split_ids(ids, tc["val_frac"])
    model = MiniGPT(GPTConfig(vocab_size=tok.vocab_size, **mc)).to(device)
    print(f"device={device} vocab={tok.vocab_size} params={model.n_params()/1e6:.3f}M "
          f"train_tokens={len(train_d)} val_tokens={len(val_d)}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    log, t0 = [], time.time()
    for step in range(tc["max_steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        if step % tc["eval_interval"] == 0:
            l = estimate_loss(model, {"train": train_d, "val": val_d}, tc, mc["context_length"], device)
            rec = {"step": step, "train_loss": l["train"], "val_loss": l["val"],
                   "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
            log.append(rec)
            print(rec)
        if step == tc["max_steps"]:
            break
        x, y = get_batch(train_d, tc["batch_size"], mc["context_length"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    wall = time.time() - t0
    ck = f"checkpoints/{a.tag}.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "chars": tok.chars}, ck)
    summary = {"tag": a.tag, "params": model.n_params(), "wall_s": wall, "device": device,
               "final_train_loss": log[-1]["train_loss"], "final_val_loss": log[-1]["val_loss"],
               "model_cfg": model.cfg.to_dict(), "train_cfg": tc, "log": log}
    with open(f"outputs/logs/{a.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {ck}; wall={wall:.0f}s final val loss={summary['final_val_loss']:.4f}")


if __name__ == "__main__":
    main()
