"""FastAPI backend. The OpenRouter key is read from the environment only and is
never sent to the browser."""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from . import cube  # noqa: E402
from .adapters import OPENROUTER_BASE  # noqa: E402
from .runner import (  # noqa: E402
    RESULTS_DIR,
    TEST_SETS,
    build_adapter,
    load_profiles,
    make_puzzles,
    manager,
)

app = FastAPI(title="Rubik's Cube Race")


class RunRequest(BaseModel):
    profile: str = "openrouter"
    test_set: str = "primary"          # warmup | primary | extension | custom
    num_puzzles: int = 5
    seed: int = 100
    max_moves: int = 30
    time_limit_s: float = 300
    custom_len: int | None = None
    models: list[str] | None = None    # subset of labels
    max_cost_usd: float = 1.0 


@app.get("/api/config")
def config():
    return {
        "profiles": load_profiles(),
        "test_sets": {k: list(v) for k, v in TEST_SETS.items()},
        "student": {
            "name": os.getenv("STUDENT_NAME", ""),
            "roll": os.getenv("STUDENT_ROLL", ""),
        },
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),  # boolean only, never the key
        "moves": cube.MOVES,
    }


@app.get("/api/puzzle")
def puzzle(test_set: str = "primary", seed: int = 1234, custom_len: int | None = None):
    return make_puzzles(test_set, 1, seed, custom_len)[0]


@app.post("/api/runs")
async def create_run(req: RunRequest):
    profiles = load_profiles()
    if req.profile not in profiles:
        raise HTTPException(400, f"unknown profile {req.profile}")
    specs = profiles[req.profile]["models"]
    if req.models:
        specs = [s for s in specs if s["label"] in req.models]
    if not specs:
        raise HTTPException(400, "no models selected")
    if req.profile == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(400, "OPENROUTER_API_KEY is not set in the backend environment (.env)")
    if req.test_set not in TEST_SETS and req.test_set != "custom":
        raise HTTPException(400, "test_set must be warmup, primary, extension or custom")
    cfg = req.model_dump()
    run = manager.create(cfg, specs)
    return {"id": run.id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = manager.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run.snapshot()


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    run = manager.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run.stop()
    return {"ok": True}


@app.post("/api/runs/{run_id}/next")
def next_benchmark(run_id: str):
    run = manager.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run.next_benchmark()
    return {"ok": True}


@app.get("/api/saved")
def saved():
    return sorted(p.name for p in RESULTS_DIR.glob("*.json"))


@app.get("/api/saved/{name}")
def saved_one(name: str):
    p = (RESULTS_DIR / name).resolve()
    if p.parent != RESULTS_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/selftest")
async def selftest(profile: str = "openrouter"):
    """One real call per model on a 3-move scramble. Use this to check your key,
    credits, and model IDs before the real benchmark."""
    profiles = load_profiles()
    if profile not in profiles:
        raise HTTPException(400, "unknown profile")
    seq, state = cube.generate_scramble(3, seed=7)
    out = []
    for spec in profiles[profile]["models"]:
        a = build_adapter(spec)
        res = await a.choose_move(state, [], use_cache=False)
        out.append({
            "label": spec["label"], "id": spec["id"], "ok": res.ok, "move": res.move,
            "latency_s": round(res.latency_s, 3), "errors": res.errors,
            "cost_usd": res.cost_usd, "confidence": res.confidence,
        })
    return out


@app.get("/api/verify-models")
async def verify_models():
    """Check the assignment model IDs against OpenRouter's live catalog."""
    wanted = {s["id"]: s for s in load_profiles()["openrouter"]["models"]}
    found = {}
    async with httpx.AsyncClient(timeout=30) as c:
        for params in ({"output_modalities": "decisions"}, {}):
            try:
                r = await c.get(OPENROUTER_BASE.rstrip("/") + "/v1/models", params=params)
                for m in r.json().get("data", []):
                    found[m["id"]] = True
            except Exception:
                pass
    return {mid: (mid in found) for mid in wanted}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
