from __future__ import annotations

import time

import pytest

from actants.cache.memory import InMemoryCache, make_key
from actants.llm.base import ChatMessage, CompletionResult, TokenUsage


def _result(text: str = "hi") -> CompletionResult:
    return CompletionResult(
        content=text,
        model="llama3.2",
        provider="ollama",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_make_key_stable():
    msgs = [
        ChatMessage(role="system", content="be nice"),
        ChatMessage(role="user", content="hi"),
    ]
    k1 = make_key(msgs, "llama3.2", 0.5)
    k2 = make_key(msgs, "llama3.2", 0.5)
    assert k1 == k2


def test_make_key_sensitive_to_temperature():
    msgs = [ChatMessage(role="user", content="hi")]
    assert make_key(msgs, "llama3.2", 0.5) != make_key(msgs, "llama3.2", 0.7)


def test_make_key_sensitive_to_model():
    msgs = [ChatMessage(role="user", content="hi")]
    assert make_key(msgs, "llama3.2", 0.5) != make_key(msgs, "llama3.3", 0.5)


@pytest.mark.asyncio
async def test_memory_cache_set_get():
    cache = InMemoryCache(default_ttl=None)
    await cache.set("k", _result("cached"))
    got = await cache.get("k")
    assert got is not None
    assert got.content == "cached"


@pytest.mark.asyncio
async def test_memory_cache_miss_returns_none():
    cache = InMemoryCache()
    assert await cache.get("nope") is None


@pytest.mark.asyncio
async def test_memory_cache_ttl_expires():
    cache = InMemoryCache(default_ttl=None)
    await cache.set("k", _result(), ttl=0)
    # ttl=0 means expires_at = now + 0
    time.sleep(0.01)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_memory_cache_clear():
    cache = InMemoryCache(default_ttl=None)
    await cache.set("a", _result())
    await cache.set("b", _result())
    assert len(cache) == 2
    await cache.clear()
    assert len(cache) == 0
