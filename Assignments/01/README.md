# MiniGPT: a causal Transformer language model from scratch

A small GPT-style language model written in PyTorch **without** pretrained weights or the `transformers` library.
It is trained on Tiny Shakespeare at the character level and exposed through a CLI and a FastAPI service.
Only tensor maths, autograd, `nn.Linear` / `nn.LayerNorm` / `nn.Embedding` and the optimiser come from PyTorch
(allowed by the assignment); attention, masking, multi-head split/merge, positional encoding, the Transformer block,
the language-model head, the loss targets and the sampling loop are written in this repository.

## Where things live

| What | Where |
|---|---|
| Model (attention, MHA, positions, block, GPT) | `src/attention.py`, `src/positional_encoding.py`, `src/transformer_block.py`, `src/gpt.py` |
| Data pipeline (tokenizer, shifted-target batches) | `src/tokenizer.py`, `src/dataset.py` |
| Training and inference | `src/train.py`, `src/generate.py`, `api.py` |
| Tests | `tests/` (attention maths, causal mask, shapes, overfit gate, tokenizer/targets/checkpoint reload) |
| Experiments and evidence figures | `experiments/` (hand calculation, visualisations, BERT-vs-GPT, ablations, evaluation, failure analysis) |
| Configs | `configs/tiny_gpt.yaml` (CPU), `configs/colab_gpu.yaml` (GPU, ~3.2M params) |
| Colab notebook (runs everything) | `MiniGPT_Assignment.ipynb` |
| Report material | `report/` (`report_template.md`, `results_cpu_run.md` real numbers from the CPU run, `demo_script.md`, `explain_every_file.md`) |
| Outputs (figures, logs, samples) | `outputs/figures`, `outputs/logs`, `outputs/samples` |

## Setup

```bash
git clone <your-repo-url> transformer-gpt-assignment && cd transformer-gpt-assignment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                       # unit tests + tiny overfit test
python -m experiments.hand_calc                           # the by-hand attention example
```

## Dataset

Tiny Shakespeare, about 1.1 MB, 65 distinct characters (<https://huggingface.co/datasets/karpathy/tiny_shakespeare>).
It is downloaded to `data/input.txt` automatically on the first training run (details in `data/README.md`).
Split: first 90% of the character stream for training, last 10% for validation. Tokenizer: character level, vocabulary =
sorted set of characters in the corpus, no special tokens, unseen characters are dropped at inference.

## Train

```bash
# CPU-friendly baseline (0.81M parameters, ~9 minutes on 2 CPU cores)
python -m src.train --config configs/tiny_gpt.yaml --tag baseline

# GPU baseline (~3.2M parameters; the one used in the Colab notebook)
python -m src.train --config configs/colab_gpu.yaml --tag baseline
```

Every evaluation step logs epoch, train loss, validation loss, learning rate, tokens/s, and a sample generated from
`ROMEO:`. The run saves `checkpoints/baseline.pt` (weights, config, vocabulary) and `outputs/logs/baseline.json`
(full summary: parameters, device, wall-clock time, losses, peak memory).

### Ablations (one factor at a time)

```bash
./run_all.sh                       # baseline + depth 2/6, heads 2/8, context 64/256, learned positions (resumable)
python -m src.train --overrides n_layers=2 --tag depth2              # a single run
python -m experiments.ablation                                       # table + Figures 6 and 9
```

For the context-length runs the batch size is rescaled so that `batch x context` (tokens per step) stays constant.

## Evaluate and produce the report figures

```bash
python -m experiments.attention_visualization   # Figs 1-5, 7 + architecture diagram
python -m experiments.mlm_vs_clm                # BERT-style vs GPT-style masking experiment
python -m experiments.seq_vs_parallel           # sequential vs parallel depth, measured
python -m experiments.evaluate                  # val loss/perplexity, samples, causal check, CLI/API evidence
python -m experiments.failure_analysis          # measured failure modes
```

## Inference

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 150 --temperature 0.8
```

Expected output shape (text differs per run and per checkpoint; this is real output from the 0.81M CPU baseline):

```
Prompt: ROMEO:
Generated: ROMEO:
Way; and my lord, fier conful, my abled down smeak,
His made to she stritess deather subban's and to for
It it a death deast the king king and them,

model parameters: 0.81M
context length: 128
temperature: 0.8
```

### REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 120, "temperature": 0.8, "seed": 1}'
```

Endpoints: `GET /health`, `POST /generate` (fields: `prompt`, `max_new_tokens`, `temperature`, `top_k`, `seed`), interactive docs at `/docs`.
Deploy on Render with `render.yaml` (it installs CPU PyTorch and starts `uvicorn api:app`); the final checkpoint
`checkpoints/baseline.pt` must be committed for the service to start. Set `CHECKPOINT` to use another file.

## Reproducibility

Seeds are fixed (`seed` in the config); runs on the same hardware repeat closely, but GPU kernels are not bit-exact.
No API keys or credentials are used anywhere; `.gitignore` excludes the dataset, virtual environments and all
checkpoints except the small final one.

## Known limitations

- Character-level model with a 128-character window: it learns spelling, line layout and speaker-name style, but invents
  non-words, repeats itself under greedy decoding, and loses long-range coherence.
- Trained on one narrow corpus, so out-of-domain prompts produce Shakespeare-flavoured nonsense.
- Short training budget and a single seed per ablation: small differences in validation loss may be noise.
- Generated text is an experimental artifact, not factual output.

## Sources and attribution

Vaswani et al. (2017) *Attention Is All You Need* (attention, sinusoidal encoding); Radford et al. (2019) GPT-2 (pre-LayerNorm
decoder block, GELU); Devlin et al. (2018) BERT (masked LM objective); Ba et al. (2016) LayerNorm; Tiny Shakespeare from
Karpathy's char-rnn. All model code in this repository was written for this assignment. Cite any additional tutorial or
repository you consult.
