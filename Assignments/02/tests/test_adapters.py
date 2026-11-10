"""Adapters tested against a fake in-process server: no network, no credits."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request

from backend import adapters, cube

fake = FastAPI()
seen = {}


@fake.post("/api/alpha/decisions")
async def decisions(req: Request):
    body = await req.json()
    seen["decisions"] = (dict(req.headers), body)
    q = body["questions"]["next_move"]
    assert q["type"] == "choice" and len(q["criteria"]) == 18
    return {"id": "x", "model": body["model"], "answers": {"next_move": {
        "type": "choice", "choice": "R'", "confidence": 0.61,
        "probabilities": {"R'": 0.61, "U": 0.39}}},
        "usage": {"input_tokens": 900, "output_tokens": 5, "cost": 0.00002}}


@fake.post("/api/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    seen["chat"] = body
    return {"choices": [{"message": {"content": json.dumps({"move": "F2"})}}], "usage": {"cost": 0.01}}


@fake.post("/api/chat")
async def ollama(req: Request):
    body = await req.json()
    seen["ollama"] = body
    return {"message": {"content": "<think>hmm</think>```json\n{\"move\": \"U'\"}\n```"}}


@fake.post("/bad/api/alpha/decisions")
async def bad_decisions(req: Request):
    return {"answers": {"next_move": {"choice": "X9"}}}


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    real = httpx.AsyncClient

    def factory(*a, **k):
        k.pop("timeout", None)
        return real(transport=httpx.ASGITransport(app=fake), base_url="http://t", timeout=5)

    monkeypatch.setattr(adapters.httpx, "AsyncClient", factory)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def run(coro):
    return asyncio.run(coro)


def test_decisions_adapter_roundtrip():
    a = adapters.OpenRouterDecisionsAdapter("typesafe/jev-1.13", "Jev", base_url="http://t/api")
    res = run(a.choose_move(cube.SOLVED, ["R", "U"]))
    assert res.ok and res.move == "R'" and res.confidence == 0.61 and res.cost_usd > 0
    headers, body = seen["decisions"]
    assert headers["authorization"] == "Bearer sk-or-test"
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"]["cube_facelets"] == cube.SOLVED
    assert body["state"]["recent_moves"] == ["R", "U"]


def test_same_question_for_all_decision_models():
    a = adapters.OpenRouterDecisionsAdapter("a/x", "A", base_url="http://t/api")
    b = adapters.OpenRouterDecisionsAdapter("b/y", "B", base_url="http://t/api")
    ra, rb = a.build_request(cube.SOLVED, []), b.build_request(cube.SOLVED, [])
    ra.pop("model"), rb.pop("model")
    assert ra == rb


def test_chat_adapter_uses_json_schema_and_parses_move():
    a = adapters.OpenRouterChatAdapter("anthropic/claude-opus-5.5", "Opus", base_url="http://t/api")
    res = run(a.choose_move(cube.SOLVED, []))
    assert res.ok and res.move == "F2"
    body = seen["chat"]
    assert body["model"] == "anthropic/claude-opus-5.5"
    assert body["response_format"]["type"] == "json_schema"
    assert set(body["response_format"]["json_schema"]["schema"]["properties"]["move"]["enum"]) == set(cube.MOVES)


def test_ollama_adapter_strips_think_and_fences():
    a = adapters.OllamaAdapter("qwen3.5", "Qwen", "Frontier LLM", base_url="http://t")
    res = run(a.choose_move(cube.SOLVED, []))
    assert res.ok and res.move == "U'"
    assert seen["ollama"]["format"]["properties"]["move"]["enum"]


def test_illegal_move_is_counted_not_converted():
    a = adapters.OpenRouterDecisionsAdapter("m/bad", "Bad", base_url="http://t/bad/api")
    res = run(a.choose_move(cube.SOLVED, []))
    assert not res.ok and res.move is None
    assert res.invalid_outputs == adapters.MAX_RETRIES + 1
    assert res.attempts == adapters.MAX_RETRIES + 1


def test_missing_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    a = adapters.OpenRouterDecisionsAdapter("m/x", "X", base_url="http://t/api")
    res = run(a.choose_move(cube.SOLVED, []))
    assert not res.ok and "OPENROUTER_API_KEY" in res.errors[-1]
    assert res.attempts == 1  # 4xx style error: no pointless retry


def test_recent_moves_window():
    a = adapters.OpenRouterDecisionsAdapter("m/x", "X", base_url="http://t/api")
    run(a.choose_move(cube.SOLVED, list("RRRRRRRRRR")))
    assert len(seen["decisions"][1]["state"]["recent_moves"]) == adapters.RECENT_MOVES_WINDOW


def test_allowed_moves_restrict_options_and_schema():
    allowed = [m for m in cube.MOVES if m != "R'"]
    a = adapters.OpenRouterDecisionsAdapter("m/x", "X", base_url="http://t/api")
    assert "R'" not in a.build_request(cube.SOLVED, [], allowed)["questions"]["next_move"]["criteria"]
    assert "R'" not in adapters._move_schema(allowed)["schema"]["properties"]["move"]["enum"]
    # a move that was not offered is rejected, not converted (fake server answers R')
    res = run(a.choose_move(cube.SOLVED, ["R"], allowed=allowed))
    assert not res.ok and res.invalid_outputs == adapters.MAX_RETRIES + 1


def test_minimal_fallback_request_is_smaller():
    a = adapters.OpenRouterDecisionsAdapter("m/x", "X", base_url="http://t/api")
    full, mini = a.build_request(cube.SOLVED, []), a.build_request(cube.SOLVED, [], minimal=True)
    assert set(mini["state"]) == {"cube_facelets", "recent_moves"}
    assert len(str(mini)) < len(str(full))
