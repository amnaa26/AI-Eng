"""Benchmark runner: puzzles, trials, live state, aggregation, persistence."""
from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
import uuid
from pathlib import Path

from . import cube
from .adapters import (
    MockAdapter,
    OllamaAdapter,
    OllamaDecisionsAdapter,
    OpenRouterChatAdapter,
    OpenRouterDecisionsAdapter,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_FILE = Path(__file__).resolve().parent / "models.json"

TEST_SETS = {
    "warmup": (3, 4),
    "primary": (4, 5),
    "extension": (15, 20),
}

MAX_CONSECUTIVE_FAILED_TURNS = 3  # same for every model


def load_profiles():
    return json.loads(MODELS_FILE.read_text())


def build_adapter(spec: dict):
    kind = spec["kind"]
    if kind == "decisions":
        return OpenRouterDecisionsAdapter(spec["id"], spec["label"], spec["category"])
    if kind == "chat":
        return OpenRouterChatAdapter(spec["id"], spec["label"], spec["category"])
    if kind == "ollama_decisions":
        return OllamaDecisionsAdapter(spec["id"], spec["label"], spec["category"])
    if kind == "ollama":
        return OllamaAdapter(spec["id"], spec["label"], spec["category"])
    if kind == "mock":
        return MockAdapter(
            spec["id"], spec["label"], spec["category"],
            latency=spec.get("latency", 0.05),
            strategy=spec.get("strategy", "random"),
            mistake_rate=spec.get("mistake_rate", 0.0),
        )
    raise ValueError(f"unknown adapter kind {kind}")


def make_puzzles(test_set: str, n: int, seed: int, custom_len: int | None = None):
    import random

    rng = random.Random(seed)
    lo, hi = TEST_SETS.get(test_set, (custom_len or 4, custom_len or 4))
    puzzles = []
    for i in range(n):
        length = rng.randint(lo, hi)
        puzzle_seed = rng.randint(0, 10**9)
        scramble, state = cube.generate_scramble(length, seed=puzzle_seed)
        opt = cube.optimal_distance(state)  # exact when <= 6, else None
        puzzles.append({
            "index": i,
            "seed": puzzle_seed,
            "scramble_length": length,
            "scramble": scramble,
            "state": state,
            "optimal": opt,
        })
    return puzzles


def _new_trial(adapter, puzzle):
    return {
        "label": adapter.label,
        "model_id": adapter.model_id,
        "category": adapter.category,
        "kind": adapter.kind,
        "status": "waiting",  # waiting | thinking | solved | not_solved | out_of_time | error | stopped
        "state": puzzle["state"],
        "moves": [],
        "log": [],
        "started_at": None,
        "finished_at": None,
        "elapsed_s": 0.0,
        "invalid_outputs": 0,
        "failed_turns": 0,
        "retries": 0,
        "cost_usd": 0.0,
        "cache_hits": 0,
        "saved_usd": 0.0,
        "errors": [],
        "optimal": puzzle["optimal"],
        "gap": None,
    }


class Run:
    def __init__(self, cfg: dict, specs: list):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.cfg = cfg
        self.specs = specs
        self.status = "created"  # created | running | waiting_next | finished | stopped | error
        self.created_at = time.time()
        self.puzzles = make_puzzles(cfg["test_set"], cfg["num_puzzles"], cfg["seed"], cfg.get("custom_len"))
        self.trials = []  # list of dict per puzzle: {label: trial}
        self.current = 0
        self.total_cost = 0.0
        self.total_saved = 0.0
        self.total_hits = 0
        self.stopped_reason = None
        self._stop = asyncio.Event()
        self._next = asyncio.Event()
        self._task = None

    # ----- snapshot --------------------------------------------------------
    def snapshot(self):
        now = time.time()
        trials_out = []
        for per_puzzle in self.trials:
            d = {}
            for label, t in per_puzzle.items():
                t = dict(t)
                if t["started_at"] and not t["finished_at"]:
                    t["elapsed_s"] = now - t["started_at"]
                d[label] = t
            trials_out.append(d)
        return {
            "id": self.id,
            "status": self.status,
            "config": self.cfg,
            "models": [{"label": s["label"], "id": s["id"], "category": s["category"], "kind": s["kind"]} for s in self.specs],
            "puzzles": self.puzzles,
            "trials": trials_out,
            "current": self.current,
            "total_cost_usd": self.total_cost,
            "cache_hits": self.total_hits,
            "saved_usd": self.total_saved,
            "stopped_reason": self.stopped_reason,
            "now": now,
            "summary": summarize(self),
        }

    # ----- execution -------------------------------------------------------
    def start(self):
        self.status = "running"
        self._task = asyncio.create_task(self._run())

    def stop(self):
        self._stop.set()
        self._next.set()

    def next_benchmark(self):
        if self.status == "waiting_next":
            self._next.set()

    async def _run(self):
        try:
            for pi, puzzle in enumerate(self.puzzles):
                if self._stop.is_set():
                    break
                self.current = pi
                adapters = [build_adapter(s) for s in self.specs]
                per = {}
                for a in adapters:
                    per[a.label] = _new_trial(a, puzzle)
                    if isinstance(a, MockAdapter):
                        a.set_plan(puzzle["scramble"])
                self.trials.append(per)
                for adapter in adapters:
                    if self._stop.is_set():
                        break
                    await self._play(adapter, per[adapter.label], puzzle)
                self.save()
                if self._stop.is_set():
                    break
                if pi < len(self.puzzles) - 1:
                    self._next.clear()
                    self.status = "waiting_next"
                    next_task = asyncio.create_task(self._next.wait())
                    stop_task = asyncio.create_task(self._stop.wait())
                    try:
                        await asyncio.wait(
                            (next_task, stop_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        next_task.cancel()
                        stop_task.cancel()
                        await asyncio.gather(next_task, stop_task, return_exceptions=True)
                    if self._stop.is_set():
                        break
                    self.status = "running"
            self.status = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:  # pragma: no cover
            self.status = "error"
            self.error = repr(e)
        finally:
            self.save()

    async def _play(self, adapter, trial, puzzle):
        max_moves = self.cfg["max_moves"]
        time_limit = self.cfg["time_limit_s"]
        state = puzzle["state"]
        trial["started_at"] = time.time()
        t0 = time.perf_counter()
        consecutive_fail = 0
        visited = {state}  # positions seen this attempt; moves back into them are not offered

        def finish(status):
            trial["status"] = status
            trial["finished_at"] = time.time()
            trial["elapsed_s"] = time.perf_counter() - t0
            trial["state"] = state
            if status == "solved" and puzzle["optimal"] is not None:
                trial["gap"] = len(trial["moves"]) - puzzle["optimal"]

        while True:
            if cube.is_solved(state):
                finish("solved")
                return
            if len(trial["moves"]) >= max_moves:
                finish("not_solved")
                return
            if self._stop.is_set():
                finish("stopped")
                return
            remaining = time_limit - (time.perf_counter() - t0)
            if remaining <= 0:
                finish("out_of_time")
                return
            trial["status"] = "thinking"
            allowed = cube.legal_moves(state, visited)
            call_task = asyncio.create_task(adapter.choose_move(state, trial["moves"], allowed=allowed))
            stop_task = asyncio.create_task(self._stop.wait())
            try:
                done, _ = await asyncio.wait(
                    (call_task, stop_task),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    call_task.cancel()
                    await asyncio.gather(call_task, return_exceptions=True)
                    finish("stopped")
                    return
                if call_task not in done:
                    call_task.cancel()
                    await asyncio.gather(call_task, return_exceptions=True)
                    finish("out_of_time")
                    return
                res = call_task.result()
            except asyncio.TimeoutError:
                finish("out_of_time")
                return
            finally:
                stop_task.cancel()
                await asyncio.gather(stop_task, return_exceptions=True)
            trial["invalid_outputs"] += res.invalid_outputs
            trial["retries"] += max(0, res.attempts - 1)
            trial["cost_usd"] += res.cost_usd
            if res.cached:
                trial["cache_hits"] += 1
                trial["saved_usd"] += res.saved_usd
                self.total_hits += 1
                self.total_saved += res.saved_usd
            self.total_cost += res.cost_usd
            cap = float(self.cfg.get("max_cost_usd") or 0)
            if cap and self.total_cost >= cap:
                self.stopped_reason = f"cost cap ${cap:.2f} reached"
                self._stop.set()
                self._next.set()
            if not res.ok:
                trial["failed_turns"] += 1
                consecutive_fail += 1
                trial["errors"].append(res.errors[-1] if res.errors else "unknown error")
                trial["log"].append({"n": len(trial["moves"]) + 1, "move": None, "latency_s": res.latency_s,
                                     "error": res.errors[-1] if res.errors else "failed"})
                if consecutive_fail >= MAX_CONSECUTIVE_FAILED_TURNS:
                    finish("error")
                    return
                continue
            consecutive_fail = 0
            state = cube.apply_move(state, res.move)
            visited.add(state)
            trial["moves"].append(res.move)
            trial["state"] = state
            trial["log"].append({
                "n": len(trial["moves"]),
                "move": res.move,
                "latency_s": round(res.latency_s, 4),
                "confidence": res.confidence,
                "cached": res.cached,
            })

    # ----- persistence -----------------------------------------------------
    def save(self):
        snap = self.snapshot()
        (RESULTS_DIR / f"{self.id}.json").write_text(json.dumps(snap, indent=1))
        rows = []
        for pi, per in enumerate(self.trials):
            p = self.puzzles[pi]
            for label, t in per.items():
                lat = [l["latency_s"] for l in t["log"] if l.get("move")]
                rows.append({
                    "puzzle": pi, "scramble_len": p["scramble_length"], "optimal": p["optimal"],
                    "model": label, "model_id": t["model_id"], "category": t["category"],
                    "status": t["status"], "moves": len(t["moves"]), "gap": t["gap"],
                    "total_time_s": round(t["elapsed_s"], 3),
                    "avg_move_latency_s": round(sum(lat) / len(lat), 4) if lat else "",
                    "invalid_outputs": t["invalid_outputs"], "failed_turns": t["failed_turns"],
                    "retries": t["retries"], "cost_usd": round(t["cost_usd"], 6),
                    "cache_hits": t["cache_hits"],
                    "move_sequence": " ".join(t["moves"]),
                })
        if rows:
            with open(RESULTS_DIR / f"{self.id}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def summarize(run: Run):
    per_model = {}
    for s in run.specs:
        per_model[s["label"]] = {"label": s["label"], "model_id": s["id"], "category": s["category"], "trials": []}
    for pi, per in enumerate(run.trials):
        for label, t in per.items():
            if t["status"] in ("waiting", "thinking"):
                continue
            per_model[label]["trials"].append((run.puzzles[pi], t))

    out = []
    for label, m in per_model.items():
        ts = m["trials"]
        n = len(ts)
        lat = [l["latency_s"] for _, t in ts for l in t["log"] if l.get("move")]
        solved = [t for _, t in ts if t["status"] == "solved"]
        gaps = [t["gap"] for _, t in ts if t["status"] == "solved" and t["gap"] is not None]
        row = {
            "label": label,
            "model_id": m["model_id"],
            "category": m["category"],
            "trials": n,
            "solved": len(solved),
            "success_rate": (100.0 * len(solved) / n) if n else None,
            "avg_move_latency_s": (sum(lat) / len(lat)) if lat else None,
            "median_move_latency_s": statistics.median(lat) if lat else None,
            "p95_move_latency_s": _pct(lat, 0.95),
            "avg_total_time_s": (sum(t["elapsed_s"] for _, t in ts) / n) if n else None,
            "avg_moves_solved": (sum(len(t["moves"]) for t in solved) / len(solved)) if solved else None,
            "avg_moves_all": (sum(len(t["moves"]) for _, t in ts) / n) if n else None,
            "avg_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "optimal_solves": sum(1 for g in gaps if g == 0),
            "gap_known_solves": len(gaps),
            "invalid_outputs": sum(t["invalid_outputs"] for _, t in ts),
            "failed_turns": sum(t["failed_turns"] for _, t in ts),
            "retries": sum(t["retries"] for _, t in ts),
            "timeouts": sum(1 for _, t in ts if t["status"] == "out_of_time"),
            "errors": sum(1 for _, t in ts if t["status"] == "error"),
            "cost_usd": sum(t["cost_usd"] for _, t in ts),
            "cache_hits": sum(t["cache_hits"] for _, t in ts),
        }
        out.append(row)
    return {"models": out, "answers": _answers(out, run)}


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _answers(rows, run):
    rows = [r for r in rows if r["trials"]]
    if not rows:
        return {}
    ans = {}
    with_lat = [r for r in rows if r["avg_move_latency_s"] is not None]
    fastest = min(with_lat, key=lambda r: r["avg_move_latency_s"]) if with_lat else None
    if fastest:
        ans["q1"] = (f"{fastest['label']} is fastest per decision "
                     f"(avg {fastest['avg_move_latency_s']*1000:.0f} ms/move).")
    best_sr = max(rows, key=lambda r: (r["success_rate"] or 0, -(r["avg_moves_solved"] or 99)))
    tied = [r["label"] for r in rows if r["success_rate"] == best_sr["success_rate"]]
    ans["q2"] = (f"{', '.join(tied)} reached the highest solve rate "
                 f"({_fmt(best_sr['success_rate'],0)}% of {best_sr['trials']} puzzles).")
    gapped = [r for r in rows if r["avg_gap"] is not None]
    if gapped:
        b = min(gapped, key=lambda r: r["avg_gap"])
        ans["q3"] = (f"{b['label']} found the shortest solutions: average optimality gap "
                     f"{_fmt(b['avg_gap'])} over {b['gap_known_solves']} solved puzzles "
                     f"({b['optimal_solves']} exactly optimal).")
    else:
        ans["q3"] = "No solved puzzle has a known exact optimum yet (exact BFS reference covers scrambles up to 6 moves)."
    if fastest and gapped:
        b = min(gapped, key=lambda r: r["avg_gap"])
        same = fastest["label"] == b["label"]
        ans["q4"] = ("Yes, the fastest model also had the best solution quality." if same else
                     f"No. {fastest['label']} was fastest per move but {b['label']} had the best solution quality.")
    elif fastest:
        ans["q4"] = f"{fastest['label']} was fastest; solution quality cannot be compared until a model solves a puzzle."
    return ans


class RunManager:
    def __init__(self):
        self.runs: dict[str, Run] = {}

    def create(self, cfg, specs) -> Run:
        run = Run(cfg, specs)
        self.runs[run.id] = run
        run.start()
        return run

    def get(self, run_id) -> Run | None:
        return self.runs.get(run_id)


manager = RunManager()
