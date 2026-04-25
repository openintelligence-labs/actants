from __future__ import annotations

from typing import Protocol

from agentic_kit.llm.base import CompletionResult


class CacheBackend(Protocol):
    async def get(self, key: str) -> CompletionResult | None: ...
    async def set(self, key: str, value: CompletionResult, ttl: int | None = None) -> None: ...
    async def clear(self) -> None: ...
