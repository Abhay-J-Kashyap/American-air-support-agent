"""Append-only JSONL cache for LLM responses.

Design notes:

* Append-only means a crashed 4-hour sweep keeps everything already written.
  Resumption is free; there is no partial-write recovery path to get wrong.
* Plain JSONL while running (~1KB/record, so a full run is single-digit MB),
  gzipped by `make freeze-cache` for committing. The committed cache is what
  lets a grader reproduce every headline number with no API key and no spend.
* Every record stores the RESOLVED provider and model id. Free-tier model
  names get retired without notice, so "which model produced this number" has
  to be answerable from the artifact, not from config at read time.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cache_key(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheRecord:
    key: str
    provider: str
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    created_at: str
    latency_ms: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class JsonlCache:
    def __init__(self, path: Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._index: dict[str, CacheRecord] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = CacheRecord(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    # A torn final line from a killed process. Skip it; the
                    # call simply re-runs.
                    continue
                self._index[rec.key] = rec

    def __len__(self) -> int:
        return len(self._index)

    def get(self, key: str) -> CacheRecord | None:
        with self._lock:
            return self._index.get(key)

    def put(self, rec: CacheRecord) -> None:
        with self._lock:
            self._index[rec.key] = rec
            if self.read_only:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")

    def stats(self) -> dict[str, int]:
        with self._lock:
            by_model: dict[str, int] = {}
            for rec in self._index.values():
                by_model[f"{rec.provider}/{rec.model}"] = (
                    by_model.get(f"{rec.provider}/{rec.model}", 0) + 1
                )
            return by_model
