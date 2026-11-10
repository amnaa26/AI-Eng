# Rubik's Cube Race

Decision models vs a frontier LLM on the same scrambled 3x3 cube, measured on **speed** (latency per move, total time)
and **solution quality** (solve rate, moves, gap to the exact shortest solution). Python + FastAPI backend, a single-page
HTML frontend, and OpenRouter: the Decisions API for Jev 1.13, GPT-6 Luna Decisions and Cloudflare Clef, and chat completions for Claude Opus 5.5.

**Headline result:** on five 4-5 move scrambles the three fast decision models solved none of them (0/15 trials), while
Claude Opus 5.5 solved 1/5 (optimally, in 4 moves) but took about 11x longer per move than the fastest decision model.

## Results: primary benchmark (5 puzzles x 4 models = 20 trials)

Scrambles of 4-5 random turns, seed `1234`, limit 30 moves and 300 s per trial, half-turn metric (18 moves).
Exact shortest solutions come from breadth-first search. Source data is in [`docs/results/`](docs/results/); see "Where the numbers come from" before relying on the table.

| Model | Type | Solved | Avg latency / move | Median | Avg total time / puzzle | Avg moves (all trials) | Avg moves (solved) | Optimality gap | Error trials | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | Decision | 0/5 (0%) | 597 ms | 604 ms | 0:18.15 | 30.0 | — | — | 0 | $0.0013 |
| GPT-6 Luna Decisions | Decision | 0/5 (0%) | 479 ms | 475 ms | 0:13.45 | 24.0 | — | — | 1 | $0.0046 |
| Clef | Decision | 0/5 (0%) | 1054 ms | 937 ms | 0:31.88 | 30.0 | — | — | 0 | $0.0211 |
| Claude Opus 5.5 | Frontier LLM | 1/5 (20%) | 5179 ms | 3729 ms | 2:10.73 | 24.8 | 4 | +0 | 0 | $1.1159 |

Failed trials use all 30 moves (reaching the limit is an unsuccessful attempt). Invalid or malformed outputs in the final run: 0 for every model.
Cost is the OpenRouter spend for the live calls behind this data set ($1.14 in total).

Per puzzle:

| # | Scramble | Optimal | Jev 1.13 | GPT-6 Luna Decisions | Clef | Claude Opus 5.5 |
|---|---|---|---|---|---|---|
| 1 | `D2 L' R U2 R` | 5 | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) |
| 2 | `L2 R' U2 D'` | 4 | ✗ not solved (30 moves) | ✗ error (HTTP 502 refusal) | ✗ not solved (30 moves) | ✓ 4 moves |
| 3 | `U' L' D' B` | 4 | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) |
| 4 | `B' L2 R U'` | 4 | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) |
| 5 | `D U F2 R' L` | 5 | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) | ✗ not solved (30 moves) |

### Answers to the experiment questions

1. **Fastest at making decisions:** GPT-6 Luna Decisions, at 479 ms per move on average. Jev 1.13 was next (597 ms), then Clef (1054 ms); Opus 5.5 averaged 5179 ms (p95 13.4 s).
2. **Highest solve rate:** Claude Opus 5.5, 1/5 (20%). Jev, GPT-6 Luna and Clef solved 0/5 each.
3. **Shortest solutions:** only Opus solved a puzzle (puzzle 2): 4 moves against an exact optimum of 4, an optimality gap of 0. This is a single solved puzzle, so it shows Opus can find an optimal solution, not that it usually does.
4. **Did the fastest model also have the best solution quality?** No. GPT-6 Luna Decisions was fastest per move but solved nothing; the slowest model produced the only solution.
5. **Fast decision model, frontier LLM, or hybrid?** *(Draft: rewrite in your own words.)* Here speed bought nothing on its own.
   The decision models answered in roughly 0.5-1 s per move, but their play had no lookahead: Jev repeated `U U'` for all 30 moves on every puzzle, and Clef and GPT-6 Luna wandered among a handful of moves.
   Opus took 2-15 s per move and cost about 41x as much as the three decision models combined, yet it was the only model to solve a puzzle.
   If wrong moves are cheap and answers are needed in milliseconds, a decision model is the practical choice; where correctness matters and a few seconds per step are acceptable, the frontier LLM is the better choice.
   A hybrid (LLM plans, a fast model executes or filters) is a reasonable design, but this experiment does not test it, so it is a hypothesis rather than a result.

### Observations and limitations

- **Small sample.** Five puzzles, one trial per model per puzzle, and one solved trial in total. Treat rankings as indicative, not statistically significant.
- **Decision models looped.** Jev and Clef alternated inverse moves or repeated one face turn. A chooser that sees only the cube and its last 6 moves has nothing that breaks such loops.
- **Opus was inconsistent.** It solved puzzle 2 optimally but failed the other four, usually by wandering and undoing its own moves until the 30-move limit. It ran with reasoning enabled, which raised latency and cost (about $0.007-$0.014 per move); the reasoning settings are in `.env.example`.
- **GPT-6 Luna refusals.** On puzzle 2's start position the endpoint answered `HTTP 502: OpenAI refused to answer question "next_move"` on every attempt in the saved runs. Each refusal is logged as a failed turn, three in a row end the trial as an error, and the result is reported as-is.
- **Opus rate limit.** OpenRouter limits new accounts to 20 requests per minute for Opus; one request hit that (HTTP 429) and the retry policy handled it.

## Where the numbers come from

The final run, `20261110-012114-6474`, shows 5 puzzles x 4 models, but **most of its moves were replayed from the response cache rather than newly requested**:

- Live run `20261110-005611-2d7a` made the real API calls (puzzles 1-4 complete and 9 Opus moves on puzzle 5) before its cost cap was reached.
- The final run replayed those recorded answers and made live Opus calls only for the remaining 21 moves of puzzle 5 (cost $0.1376).
- Cached moves keep their **original recorded latency** (the app waits that long on a hit) and are flagged `cached` in the JSON, CSV, move chips and slide.
- Some cache hits happened inside the live run itself: when a model faced an identical position with the same last-6-move history (Jev's `U U'` loop) the identical request was answered from cache. Those latencies are repeats of one earlier measurement, so Jev has only 40 independent live latency samples for 150 moves. Live-sample averages are close to the table: Jev 617 ms, GPT-6 Luna 478 ms, Clef 1054 ms, Opus 5179 ms.
- The cache holds only successful, valid answers. Failures such as the Luna refusals are never cached and were requested live in each run.
- The repository kept about two dozen earlier debugging runs (connection errors before Ollama was reachable, empty Opus replies when reasoning used up the token cap, out-of-credit 402 and rate-limit 429 errors). They are not part of this data set.
- Across all saved OpenRouter runs, recorded spend was about $3.26; this data set itself cost $1.14. Before the cost-tracking fix, early failed Opus calls were billed but recorded as $0.

Evidence in `docs/results/`: both run JSON files with CSV exports, and `response_cache.jsonl`, which holds every recorded model answer (model, move, latency, cost) so the replay can be reproduced offline.

## Setup

```bash
python -m venv .venv            # optional
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env ; then set STUDENT_NAME / STUDENT_ROLL
python -m uvicorn backend.main:app --port 8000
```

Open http://localhost:8000. Run the tests with `python -m pytest -q tests`. The OpenRouter key goes only in `.env` (a backend environment variable); it is never sent to the browser, and `.env` is git-ignored.

## Running a benchmark

1. **Mock** set: UI and pipeline check; no network, free.
2. **Ollama** set (free, local): exercises the real HTTP/JSON path. Pull the models listed in `backend/models.json` (`ollama` section) and match the tags to `ollama list`.
   Decision models such as `clef-flash` are served on Ollama's `/v1/systemone`; LLMs use `/api/chat` with JSON-constrained output. For debugging only, not reported results.
3. **OpenRouter** set: the real benchmark. Add credits (pay-as-you-go; one balance covers every model), put the key in `.env`, then run
   **Self-test models** (about 4 calls; checks key, credits, model IDs and cost), a **Warm-up** with 1 puzzle, and the **Primary** run with 5+ puzzles.

Controls: **Cost cap ($)** stops a run when total spend reaches it; **Time limit**, **Max moves** and **Seed** apply per run.
Settings in `.env`: `CHAT_MAX_TOKENS`, `CHAT_REASONING_JSON` (for example `{"effort":"none"}` to turn Opus reasoning off, or `{"effort":"low"}`),
`RESPONSE_CACHE=off` for a fully live run, `CACHE_REPLAY_LATENCY`, `REQUEST_TIMEOUT_S` and `MAX_RETRIES`.

## How it works

- `backend/cube.py`: deterministic engine (54-facelet state, 18 moves in the half-turn metric, seeded scrambles) and an exact shortest-solution search
  (meet-in-the-middle BFS, exact up to 6 moves). Optimality gap = model moves - exact shortest moves. The extension set (15-20 turns) has no exact optimum, so no gap is reported there.
- `backend/adapters.py`: one game-action contract for every model. Decision models use `POST /api/alpha/decisions` with a `choice` question whose criteria are the 18 moves
  (identical wording for all three; only `model` changes) and record `choice`, `confidence`, `probabilities` and `usage.cost`. Opus uses `POST /api/v1/chat/completions` with a strict JSON schema `{"move": ...}`.
  All models get the same timeout, one retry and the last 6 moves as history. Illegal or malformed outputs are counted, never converted into a move; 3 failed turns in a row end a trial as an error.
  Billed-but-unusable responses (such as an empty reply that used up the token cap) still have their cost counted.
- `backend/cache.py`: on-disk response cache keyed by model, state, recent moves, prompt wording and reasoning settings; successes only; flagged everywhere.
- `backend/runner.py`: models play each puzzle one after another; reaching the move limit (30) or time limit (300 s) is an unsuccessful attempt; saves `results/<run>.json` and `.csv`; enforces the cost cap.
- Frontend: live race cards and statistics table, then an auto-opening benchmark slide with per-model statistics, answers to questions 1-4 and per-puzzle results.

## Recording checklist

- Student name and roll number sit in the fixed top banner (set in `.env` or **edit**); keep them in frame throughout.
- Show the in-progress stage (cards "thinking"), the completion stage (SOLVED / RAN OUT OF MOVES / OUT OF TIME), and end on the benchmark slide.
- If you record a cache replay instead of live calls, say so: the app marks cached moves, and "Where the numbers come from" explains how the data was produced.

