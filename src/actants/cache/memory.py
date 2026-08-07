from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from actants.cache.request import KEY_VERSION as _KEY_VERSION  # noqa: F401 — re-export
from actants.cache.request import CacheRequest

if TYPE_CHECKING:
    from actants.llm.base import ChatMessage, CompletionResult, ToolSpec


def make_key(
    messages: list[ChatMessage],
    model: str,
    temperature: float,
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    tools: list[ToolSpec] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Hash every request parameter that can change the answer.

    Thin wrapper over `CacheRequest.key`; kept because it is the documented way to
    build a key by hand. New code should build a `CacheRequest` and call
    ``.key()`` on it, which is also what the cache protocol passes to backends.
    """
    return CacheRequest(
        messages=messages,
        model=model,
        temperature=temperature,
        provider=provider,
        max_tokens=max_tokens,
        tools=tools,
        response_format=response_format,
    ).key()


class InMemoryCache:
    """Exact-match cache. For semantic caching, use SqliteVecCache (optional extra)."""

    def __init__(self, *, default_ttl: int | None = 3600) -> None:
        self._data: dict[str, tuple[CompletionResult, float | None]] = {}
        #: Public to match `SqliteVecCache`, the other
        #: reference backend — third parties copy whichever they read first, so the two
        #: must agree.
        self.default_ttl = default_ttl

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
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl is not None else None
        self._data[key] = (value, expires_at)

    async def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
