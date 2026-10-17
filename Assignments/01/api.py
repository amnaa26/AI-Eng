"""FastAPI wrapper around the trained MiniGPT.

Run:   uvicorn api:app --host 0.0.0.0 --port 8000
Try:   curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
            -d '{"prompt": "ROMEO:", "max_new_tokens": 120, "temperature": 0.8, "seed": 1}'
"""
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.generate import generate_text, load_model
from src.utils import set_seed

CKPT = os.environ.get("CHECKPOINT", "checkpoints/baseline.pt")
MAX_TOKENS = 1000

app = FastAPI(title="MiniGPT", description="Character-level causal Transformer trained from scratch on Tiny Shakespeare")
model, tok = load_model(CKPT)
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


class GenerateRequest(BaseModel):
    prompt: str = "ROMEO:"
    max_new_tokens: int = Field(200, ge=1, le=MAX_TOKENS)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=1)
    seed: int | None = None          # set for reproducible output


@app.get("/")
def root():
    return {"service": "MiniGPT", "endpoints": ["/health", "/generate"], "docs": "/docs"}


@app.get("/health")
def health():
    c = model.cfg
    return {"status": "ok", "params": model.n_params(), "context_length": c.context_length,
            "d_model": c.d_model, "n_layers": c.n_layers, "n_heads": c.n_heads, "vocab_size": c.vocab_size}


@app.post("/generate")
def generate(r: GenerateRequest):
    if r.seed is not None:
        set_seed(r.seed)
    text = generate_text(model, tok, r.prompt, r.max_new_tokens, r.temperature, r.top_k)
    return {"prompt": r.prompt, "generated": text, "temperature": r.temperature,
            "seed": r.seed, "context_length": model.cfg.context_length}
