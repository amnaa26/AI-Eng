"""On-disk response cache for paid model calls.

Key = sha256 of (model, endpoint kind, cube state, recent moves, prompt version,
reasoning settings). Only successful, valid answers are stored. A hit costs
nothing; the original latency and cost are stored so results stay honest:
by default a hit sleeps for the original latency (CACHE_REPLAY_LATENCY=on) and
is flagged as `cached` everywhere (logs, JSON, CSV, UI).

Env: RESPONSE_CACHE=on|off (default on), CACHE_REPLAY_LATENCY=on|off (default on).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

CACHE_FILE = Path(__file__).resolve().parent.parent / "results" / "response_cache.jsonl"


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


def enabled() -> bool:
    return _flag("RESPONSE_CACHE", "on")


def replay_latency() -> bool:
    return _flag("CACHE_REPLAY_LATENCY", "on")


def make_key(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        self._loaded = False

    def _load(self):
        self._loaded = True
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                self.data[row["key"]] = row
            except Exception:
                continue  # ignore a corrupt line

    def get(self, key: str):
        if not self._loaded:
            self._load()
        return self.data.get(key)

    def put(self, key: str, entry: dict):
        if not self._loaded:
            self._load()
        row = dict(entry, key=key, ts=time.time())
        self.data[key] = row
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass  # a read-only disk must never break a run

    def __len__(self):
        if not self._loaded:
            self._load()
        return len(self.data)


cache = ResponseCache(CACHE_FILE)
