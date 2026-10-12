# MiniGPT: a Transformer language model from scratch

Causal decoder-only Transformer in PyTorch trained on Tiny Shakespeare (character-level). No pretrained weights, no `transformers` library.

## Layout
- `src/attention.py`: scaled dot-product attention, causal mask, multi-head attention
- `src/positional_encoding.py`: sinusoidal encoding
- `src/transformer_block.py`: LayerNorm + attention + GELU FFN with residuals
- `src/gpt.py`: MiniGPT (embedding, blocks, LM head, generation)
- `src/tokenizer.py`, `src/dataset.py`: char tokenizer, shifted-target batches
- `src/train.py`, `src/generate.py`, `api.py`: training, CLI inference, FastAPI
- `tests/`: unit tests (attention, mask/causality, shapes, overfit)
- `experiments/`: hand calculation, plus (to add) ablation, attention visualization

## Setup
```
pip install -r requirements.txt
pytest -q
python -m experiments.hand_calc
```

## Train (downloads Tiny Shakespeare into data/input.txt)
```
python -m src.train --config configs/tiny_gpt.yaml --tag baseline
```
Ablation example: `python -m src.train --overrides n_layers=2 --tag depth2`

## Inference
```
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
uvicorn api:app --port 8000
curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt":"ROMEO:"}'
```

## Dataset
Tiny Shakespeare (about 1.1 MB), https://huggingface.co/datasets/karpathy/tiny_shakespeare

## Known limitations
Character-level model with a short context; generated text imitates style but is not factually reliable. (Extend after your experiments.)
