from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actants.llm.base import ChatMessage, CompletionResult, ToolSpec

#: Bump when the key layout changes so old entries can never be misread as new ones.
_KEY_VERSION = 2


def make_key(
    messages: list[ChatMessage],
    model: str,
    temperature: float,
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    tools: list[ToolSpec] | None = None,
) -> str:
    """Hash every request parameter that can change the answer.

    The key covers provider, model, temperature, ``max_tokens``, the tool definitions,
    and the full message list (including tool-call structure). Anything omitted here
    becomes a cache collision — two different requests silently sharing one answer.

    The payload is serialized as canonical JSON rather than concatenated bytes so that
    message content can never forge a field boundary.
    """
    payload = {
        "v": _KEY_VERSION,
        "provider": provider,
        "model": model,
        # Round to the precision providers actually honour, but keep enough digits that
        # distinct temperatures stay distinct.
        "temperature": round(float(temperature), 6),
        "max_tokens": max_tokens,
        "tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in (tools or [])
        ],
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "name": m.name,
                "tool_call_id": m.tool_call_id,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
                ],
            }
            for m in messages
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


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
