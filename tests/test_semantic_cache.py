from __future__ import annotations

import math

import pytest

pytest.importorskip("sqlite_vec")

from actants.cache.semantic import SqliteVecCache  # noqa: E402
from actants.llm.base import ChatMessage, CompletionResult, TokenUsage  # noqa: E402


class StubEmbedder:
    """Deterministic tiny embedder. Similar strings get similar vectors."""

    async def embed(self, text: str) -> list[float]:
        buckets = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            buckets[i % 8] += ord(ch)
        norm = math.sqrt(sum(b * b for b in buckets)) or 1.0
        return [b / norm for b in buckets]


def _result(text: str = "hi") -> CompletionResult:
    return CompletionResult(
        content=text,
        model="llama3.2",
        provider="ollama",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.asyncio
async def test_sqlite_vec_cache_roundtrip(tmp_path):
    cache = SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=0.5,
        default_ttl=None,
    )
    msgs = [ChatMessage(role="user", content="what is the capital of france")]
    await cache.set_by_messages(msgs, "llama3.2", 0.7, _result("Paris"))

    hit = await cache.get_by_messages(msgs, "llama3.2", 0.7)
    assert hit is not None
    assert hit.content == "Paris"
    cache.close()


@pytest.mark.asyncio
async def test_sqlite_vec_cache_different_model_misses(tmp_path):
    cache = SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=0.5,
        default_ttl=None,
    )
    msgs = [ChatMessage(role="user", content="hello")]
    await cache.set_by_messages(msgs, "model-a", 0.7, _result())
    miss = await cache.get_by_messages(msgs, "model-b", 0.7)
    assert miss is None
    cache.close()


@pytest.mark.asyncio
async def test_sqlite_vec_cache_threshold_rejects_dissimilar(tmp_path):
    cache = SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=0.0001,  # extremely strict
        default_ttl=None,
    )
    await cache.set_by_messages(
        [ChatMessage(role="user", content="hello world")],
        "m",
        0.7,
        _result(),
    )
    # Very different query shouldn't be returned under strict threshold
    miss = await cache.get_by_messages(
        [ChatMessage(role="user", content="completely unrelated text xyzzy")],
        "m",
        0.7,
    )
    assert miss is None
    cache.close()


@pytest.mark.asyncio
async def test_semantic_cache_via_llm_client(tmp_path):
    from collections.abc import AsyncIterator

    from actants.llm.base import BaseLLMProvider
    from actants.llm.client import LLM

    class CountingProvider(BaseLLMProvider):
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
            self.calls += 1
            return _result(f"reply-{self.calls}")

        async def stream(
            self, messages, model, temperature=0.7, max_tokens=None, **kwargs
        ) -> AsyncIterator[str]:
            yield ""

        async def health(self) -> bool:
            return True

    provider = CountingProvider()
    cache = SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=0.5,
        default_ttl=None,
    )
    llm = LLM(provider=provider, model="llama3.2", cache=cache, tracing=False)

    r1 = await llm.complete("capital of france")
    r2 = await llm.complete("capital of france")
    assert r1.content == r2.content
    assert provider.calls == 1  # second call served from semantic cache
    cache.close()


@pytest.mark.asyncio
async def test_sqlite_vec_cache_clear(tmp_path):
    cache = SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=1.0,
        default_ttl=None,
    )
    await cache.set_by_messages([ChatMessage(role="user", content="a")], "m", 0.7, _result())
    assert len(cache) == 1
    await cache.clear()
    assert len(cache) == 0
    cache.close()
