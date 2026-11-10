"""Model adapters.

Every adapter implements the same game-action contract:

    await adapter.choose_move(state, recent_moves) -> MoveResult

`state` is the 54-character facelet string and `recent_moves` the last few
moves actually applied (same history policy for every model). The adapter
returns exactly one move from cube.MOVES, or a failed result (never a silently
substituted move). Timeout and retry policy are identical for all adapters.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from . import cache as rcache
from . import cube

OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Same policy for every model.
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "90"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))  # extra attempts after the first
RECENT_MOVES_WINDOW = 6
# Give every model the position that results from each candidate move, so no
# model has to simulate cube rotations in its head. Same info for all models.
LOOKAHEAD = os.getenv("LOOKAHEAD", "1") != "0"

MOVE_DESCRIPTIONS = {}
_FACE_NAMES = {
    "U": "upper",
    "R": "right",
    "F": "front",
    "D": "lower",
    "L": "left",
    "B": "back",
}
for _m in cube.MOVES:
    _f = _FACE_NAMES[_m[0]]
    if _m.endswith("2"):
        MOVE_DESCRIPTIONS[_m] = f"Turn the {_f} face by 180 degrees"
    elif _m.endswith("'"):
        MOVE_DESCRIPTIONS[_m] = f"Turn the {_f} face counterclockwise by 90 degrees"
    else:
        MOVE_DESCRIPTIONS[_m] = f"Turn the {_f} face clockwise by 90 degrees"

QUESTION = (
    "Choose the single next move most likely to progress toward the solved cube. "
    "The solved cube has every face a single colour; each sticker letter names the "
    "face it belongs to when solved. Only the listed options are legal this turn: "
    "moves that would return to a position already visited have been removed."
)

MINIMAL_QUESTION = "Pick the next move that gets this 3x3 Rubik's Cube closer to solved."

LAYOUT_NOTE = (
    "54 stickers in order U,R,F,D,L,B faces (9 each, row by row as seen looking at "
    "that face; U face top row is next to B, D face top row is next to F). "
    "Each letter is the face the sticker belongs to. Solved = UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB."
)


@dataclass
class MoveResult:
    ok: bool
    move: Optional[str] = None
    latency_s: float = 0.0          # latency of the accepted/last call
    total_call_time_s: float = 0.0  # includes retries
    attempts: int = 1
    invalid_outputs: int = 0        # malformed / illegal responses seen this turn
    errors: list = field(default_factory=list)
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    cost_usd: float = 0.0
    raw: Optional[str] = None
    cached: bool = False            # replayed from the response cache (no API call, no cost)
    saved_usd: float = 0.0          # what the original call cost (saved by the hit)


class AdapterError(Exception):
    pass


def state_payload(state: str, recent: list, allowed=None, minimal: bool = False) -> dict:
    """The one state description shared by ALL models (decisions + chat)."""
    if minimal:
        return {"cube_facelets": state, "recent_moves": recent}
    allowed = list(allowed or cube.MOVES)
    p = {
        "cube_facelets": state,
        "faces": cube.face_grids(state),
        "stickers_correct": f"{cube.correct_stickers(state)}/54",
        "recent_moves": recent,
        "facelet_layout": LAYOUT_NOTE,
    }
    if LOOKAHEAD:
        p["after_each_move"] = {
            m: {"cube_facelets": (ns := cube.apply_move(state, m)),
                "stickers_correct": cube.correct_stickers(ns)}
            for m in allowed
        }
    return p


def _criteria(allowed=None) -> dict:
    return {m: MOVE_DESCRIPTIONS[m] for m in (allowed or cube.MOVES)}


class InvalidOutput(AdapterError):
    """Response arrived but is unusable. May still have been billed (cost)."""

    def __init__(self, msg="", cost=0.0):
        super().__init__(msg)
        self.cost = cost or 0.0


class ModelRefusal(InvalidOutput):
    """The model declined to answer (e.g. OpenAI 'refused to answer question')."""


@functools.lru_cache(maxsize=1)
def prompt_signature() -> str:
    """Changes whenever the wording of the question/moves changes, so stale
    cached answers are never reused for a different prompt."""
    blob = json.dumps([QUESTION, LAYOUT_NOTE, MOVE_DESCRIPTIONS, RECENT_MOVES_WINDOW, LOOKAHEAD,
                       _chat_prompt(cube.SOLVED, [])], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class BaseAdapter:
    kind = "base"
    cacheable = True

    def __init__(self, model_id: str, label: str, category: str):
        self.model_id = model_id
        self.label = label
        self.category = category

    async def _call_once(self, client: httpx.AsyncClient, state: str, recent: list, allowed: list, attempt: int = 1):
        """Return (move, extras dict, raw text). Raise InvalidOutput / AdapterError."""
        raise NotImplementedError

    def cache_signature(self, state: str, recent: list, allowed: list) -> dict:
        return {"kind": self.kind, "model": self.model_id, "state": state,
                "recent": recent, "allowed": allowed, "prompt": prompt_signature()}

    async def choose_move(self, state: str, recent_moves: list, use_cache: bool = True,
                          allowed=None) -> MoveResult:
        recent = list(recent_moves)[-RECENT_MOVES_WINDOW:]
        allowed = [m for m in cube.MOVES if m in set(allowed or cube.MOVES)]
        key = None
        if use_cache and self.cacheable and rcache.enabled():
            key = rcache.make_key(self.cache_signature(state, recent, allowed))
            hit = rcache.cache.get(key)
            if hit and hit.get("move", "") in allowed:
                lat = float(hit.get("latency_s") or 0.0)
                if rcache.replay_latency() and lat > 0:
                    await asyncio.sleep(min(lat, REQUEST_TIMEOUT_S))
                return MoveResult(
                    ok=True, move=hit["move"], latency_s=lat, total_call_time_s=lat, attempts=1,
                    confidence=hit.get("confidence"), probabilities=hit.get("probabilities"),
                    cost_usd=0.0, cached=True, saved_usd=float(hit.get("cost") or 0.0),
                )
        res = MoveResult(ok=False)
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            for attempt in range(1, MAX_RETRIES + 2):
                t0 = time.perf_counter()
                try:
                    move, extras, raw = await self._call_once(client, state, recent, allowed, attempt)
                    dt = time.perf_counter() - t0
                    res.total_call_time_s += dt
                    res.attempts = attempt
                    res.raw = raw
                    if move not in allowed:
                        res.cost_usd += extras.get("cost", 0.0) or 0.0
                        res.invalid_outputs += 1
                        why = "illegal" if not cube.is_valid_move(move) else "not an offered"
                        res.errors.append(f"{why} move {move!r}")
                        continue
                    res.ok = True
                    res.move = move
                    res.latency_s = dt
                    res.confidence = extras.get("confidence")
                    res.probabilities = extras.get("probabilities")
                    res.cost_usd += extras.get("cost", 0.0) or 0.0
                    if key:
                        rcache.cache.put(key, {
                            "model": self.model_id, "move": move, "latency_s": dt,
                            "confidence": res.confidence, "probabilities": res.probabilities,
                            "cost": extras.get("cost", 0.0) or 0.0,
                        })
                    return res
                except InvalidOutput as e:
                    dt = time.perf_counter() - t0
                    res.total_call_time_s += dt
                    res.attempts = attempt
                    res.invalid_outputs += 1
                    res.cost_usd += e.cost
                    res.errors.append(f"invalid output: {e}")
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    dt = time.perf_counter() - t0
                    res.total_call_time_s += dt
                    res.attempts = attempt
                    res.errors.append(f"network/timeout: {type(e).__name__}")
                except AdapterError as e:
                    dt = time.perf_counter() - t0
                    res.total_call_time_s += dt
                    res.attempts = attempt
                    res.errors.append(str(e))
                    if "HTTP 4" in str(e) and "HTTP 429" not in str(e) and "HTTP 408" not in str(e):
                        break  # client error: retrying will not help
        res.latency_s = res.total_call_time_s
        return res


def _extract_json(text: str):
    """Pull a JSON object out of text (strips <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise InvalidOutput(f"no JSON object in response: {text[:120]!r}")


def _http_error(resp: httpx.Response) -> AdapterError:
    return AdapterError(f"HTTP {resp.status_code}: {resp.text[:300]}")


# ---------------------------------------------------------------------------
# OpenRouter Decisions API  (System-1 decision models)
# ---------------------------------------------------------------------------
class OpenRouterDecisionsAdapter(BaseAdapter):
    kind = "decisions"

    def __init__(self, model_id, label, category="System-1 Decision model", base_url=None, api_key_env="OPENROUTER_API_KEY"):
        super().__init__(model_id, label, category)
        self.url = (base_url or OPENROUTER_BASE).rstrip("/") + "/alpha/decisions"
        self.api_key_env = api_key_env

    def build_request(self, state, recent, allowed=None, minimal=False):
        """`minimal` is the fallback sent after a refusal: bare state, short
        wording. Applied identically to every decision model."""
        return {
            "model": self.model_id,
            "state": state_payload(state, recent, allowed, minimal),
            "questions": {
                "next_move": {
                    "type": "choice",
                    "instructions": MINIMAL_QUESTION if minimal else QUESTION,
                    "criteria": _criteria(allowed),
                }
            },
        }

    def _headers(self):
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise AdapterError("HTTP 401: OPENROUTER_API_KEY is not set")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "Rubiks Cube Race",
        }

    async def _call_once(self, client, state, recent, allowed, attempt=1):
        r = await client.post(
            self.url,
            headers=self._headers(),
            json=self.build_request(state, recent, allowed, minimal=attempt > 1),
        )
        if r.status_code >= 400:
            if "refused" in r.text.lower():
                raise ModelRefusal(f"HTTP {r.status_code} refusal: {r.text[:160]}")
            raise _http_error(r)
        try:
            data = r.json()
            ans = data["answers"]["next_move"]
            move = ans["choice"]
        except Exception:
            raise InvalidOutput(f"unexpected response shape: {r.text[:200]!r}")
        extras = {
            "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"),
            "cost": (data.get("usage") or {}).get("cost", 0.0),
        }
        return str(move).strip(), extras, json.dumps(ans)


class OllamaDecisionsAdapter(OpenRouterDecisionsAdapter):
    """Local System-1 decision model (e.g. clef-flash) served by Ollama.

    Ollama serves decision models on POST /v1/systemone (not /api/chat), with
    the same request/response shape as the OpenRouter Decisions API. No key, no cost.
    """
    kind = "ollama_decisions"

    def __init__(self, model_id, label, category="System-1 Decision model", base_url=None):
        BaseAdapter.__init__(self, model_id, label, category)
        self.url = (base_url or OLLAMA_BASE).rstrip("/") + "/v1/systemone"

    def _headers(self):
        return {"Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# OpenRouter chat completions (frontier LLM)
# ---------------------------------------------------------------------------
def _chat_prompt(state, recent, allowed=None):
    allowed = list(allowed or cube.MOVES)
    system = (
        "You are playing a 3x3 Rubik's Cube. You receive the cube as a 54-character "
        "facelet string and must output exactly ONE next move that makes progress toward "
        "the solved cube. Plan with the cube geometry, but never output a move that is not "
        "in the allowed list. Reply with JSON only, e.g. {\"move\": \"R'\"}."
    )
    moves_txt = "\n".join(f"{m}: {MOVE_DESCRIPTIONS[m]}" for m in allowed)
    p = state_payload(state, recent, allowed)
    grids = "\n".join(f"{f}: {' / '.join(rows)}" for f, rows in p["faces"].items())
    user = (
        f"{QUESTION}\n\n"
        f"Facelet layout: {LAYOUT_NOTE}\n\n"
        f"Current state: {state}\n"
        f"Faces (3 rows each): \n{grids}\n"
        f"Stickers in place: {p['stickers_correct']}\n"
        f"Recent moves (oldest to newest): {json.dumps(recent)}\n\n"
        f"Allowed moves:\n{moves_txt}\n\n"
    )
    if LOOKAHEAD:
        look = "\n".join(f"{m}: {v['cube_facelets']} ({v['stickers_correct']}/54)"
                         for m, v in p["after_each_move"].items())
        user += f"Position after each allowed move:\n{look}\n\n"
    user += 'Return exactly one move as JSON: {"move": "<one of the allowed moves>"}'
    return system, user


def _move_schema(allowed=None):
    return {
        "name": "cube_move",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"move": {"type": "string", "enum": list(allowed or cube.MOVES)}},
            "required": ["move"],
            "additionalProperties": False,
        },
    }


MOVE_SCHEMA = _move_schema()


class OpenRouterChatAdapter(BaseAdapter):
    kind = "chat"

    def cache_signature(self, state, recent, allowed):
        sig = super().cache_signature(state, recent, allowed)
        sig["reasoning"] = os.getenv("CHAT_REASONING_JSON", "").strip()
        return sig

    def __init__(self, model_id, label, category="Frontier LLM", base_url=None, api_key_env="OPENROUTER_API_KEY"):
        super().__init__(model_id, label, category)
        self.url = (base_url or OPENROUTER_BASE).rstrip("/") + "/v1/chat/completions"
        self.api_key_env = api_key_env

    async def _call_once(self, client, state, recent, allowed, attempt=1):
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise AdapterError("HTTP 401: OPENROUTER_API_KEY is not set")
        system, user = _chat_prompt(state, recent, allowed)
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "4096")),
            "response_format": {"type": "json_schema", "json_schema": _move_schema(allowed)},
            "usage": {"include": True},
        }
        # Optional OpenRouter `reasoning` object, e.g. {"effort":"none"} to switch
        # thinking off, or {"max_tokens":1024}. Unset = send nothing (model default).
        reasoning = os.getenv("CHAT_REASONING_JSON", "").strip()
        if reasoning:
            try:
                body["reasoning"] = json.loads(reasoning)
            except ValueError:
                raise AdapterError("HTTP 400: CHAT_REASONING_JSON in .env is not valid JSON")
        r = await client.post(
            self.url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "Rubiks Cube Race",
            },
            json=body,
        )
        if r.status_code >= 400:
            raise _http_error(r)
        data = r.json()
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
        except Exception:
            raise InvalidOutput(f"unexpected response shape: {r.text[:200]!r}")
        usage = data.get("usage") or {}
        cost = usage.get("cost", 0.0) or 0.0
        if not content.strip():
            finish = (data["choices"][0] or {}).get("finish_reason")
            rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            raise InvalidOutput(
                f"empty reply (finish_reason={finish}, output_tokens={usage.get('completion_tokens')}, "
                f"reasoning_tokens={rt}) - the token cap was probably used up by reasoning", cost)
        try:
            obj = _extract_json(content)
        except InvalidOutput as e:
            raise InvalidOutput(str(e), cost)
        if not isinstance(obj, dict) or "move" not in obj:
            raise InvalidOutput(f"missing 'move' key: {content[:120]!r}", cost)
        return str(obj["move"]).strip(), {"cost": cost}, content


# ---------------------------------------------------------------------------
# Ollama (local models, for testing before spending OpenRouter credits)
# ---------------------------------------------------------------------------
class OllamaAdapter(BaseAdapter):
    kind = "ollama"

    def __init__(self, model_id, label, category, base_url=None):
        super().__init__(model_id, label, category)
        self.url = (base_url or OLLAMA_BASE).rstrip("/") + "/api/chat"

    async def _call_once(self, client, state, recent, allowed, attempt=1):
        system, user = _chat_prompt(state, recent, allowed)
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": _move_schema(allowed)["schema"],
            "options": {"temperature": 0},
        }
        r = await client.post(self.url, json=body)
        if r.status_code >= 400:
            raise _http_error(r)
        content = (r.json().get("message") or {}).get("content", "")
        obj = _extract_json(content)
        if not isinstance(obj, dict) or "move" not in obj:
            raise InvalidOutput(f"missing 'move' key: {content[:120]!r}")
        return str(obj["move"]).strip(), {}, content


# ---------------------------------------------------------------------------
# Mock adapters (offline demo / tests; no network, no cost)
# ---------------------------------------------------------------------------
class MockAdapter(BaseAdapter):
    """Plays a configurable strategy with simulated latency.

    strategy "optimal-ish": replays the inverse of the scramble (needs the
    scramble via set_plan). strategy "random": picks random legal moves.
    """
    kind = "mock"
    cacheable = False

    def __init__(self, model_id, label, category, latency=0.05, strategy="random", mistake_rate=0.0, seed=0):
        super().__init__(model_id, label, category)
        self.latency = latency
        self.strategy = strategy
        self.mistake_rate = mistake_rate
        self.rng = random.Random(seed)
        self.plan = []

    def set_plan(self, scramble):
        self.plan = [cube.inverse_move(m) for m in reversed(scramble)]

    async def _call_once(self, client, state, recent, allowed, attempt=1):
        await asyncio.sleep(self.latency * (0.7 + 0.6 * self.rng.random()))
        if self.strategy == "plan" and self.plan and self.rng.random() >= self.mistake_rate:
            move = self.plan.pop(0)
            if move not in allowed:
                move = self.rng.choice(allowed)
        else:
            move = self.rng.choice(allowed)
        return move, {"confidence": round(self.rng.random(), 2)}, json.dumps({"move": move})
