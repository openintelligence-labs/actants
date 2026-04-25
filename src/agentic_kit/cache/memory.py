from __future__ import annotations

import hashlib
import time

from agentic_kit.llm.base import ChatMessage, CompletionResult


def make_key(messages: list[ChatMessage], model: str, temperature: float) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(f"{temperature:.3f}".encode())
    for m in messages:
        h.update(m.role.encode())
        h.update(b"\x00")
        h.update(m.content.encode())
        h.update(b"\x01")
    return h.hexdigest()


class InMemoryCache:
    """Exact-match cache. For semantic caching, use SqliteVecCache (optional extra)."""

    def __init__(self, default_ttl: int | None = 3600) -> None:
        self._data: dict[str, tuple[CompletionResult, float | None]] = {}
        self._default_ttl = default_ttl

    async def get(self, key: str) -> CompletionResult | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            self._data.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: CompletionResult, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl is not None else None
        self._data[key] = (value, expires_at)

    async def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
