"""Generate text from a checkpoint. Usage:
    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
"""
import argparse
import torch
from .tokenizer import CharTokenizer
from .gpt import GPTConfig, MiniGPT


def load_model(path="checkpoints/baseline.pt", device="cpu"):
    ck = torch.load(path, map_location=device)
    model = MiniGPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), CharTokenizer(ck["chars"])


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    ids = torch.tensor([tok.encode(prompt) or [0]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature, top_k)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    a = ap.parse_args()
    model, tok = load_model(a.checkpoint)
    text = generate_text(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k)
    print(f"Prompt: {a.prompt}\nGenerated: {text}\n")
    print(f"model parameters: {model.n_params()/1e6:.2f}M")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {a.temperature}")


if __name__ == "__main__":
    main()
