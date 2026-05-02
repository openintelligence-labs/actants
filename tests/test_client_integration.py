from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from actants.cache.memory import InMemoryCache
from actants.cost.tracker import CostTracker
from actants.llm.base import (
    BaseLLMProvider,
    CompletionResult,
    TokenUsage,
)
from actants.llm.client import LLM
from actants.policies.retry import RetryPolicy


class CountingProvider(BaseLLMProvider):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        self.calls += 1
        return CompletionResult(
            content=f"reply-{self.calls}",
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost_usd=0.001,
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


class FailThenSucceedProvider(BaseLLMProvider):
    name = "flaky"

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls = 0

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("transient")
        return CompletionResult(
            content="ok",
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_cache_hit_skips_provider_call():
    provider = CountingProvider()
    cache = InMemoryCache(default_ttl=None)
    llm = LLM(provider=provider, model="m", cache=cache, tracing=False)
    r1 = await llm.complete("ask")
    r2 = await llm.complete("ask")
    assert r1.content == r2.content == "reply-1"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_cost_tracker_records_per_tag():
    provider = CountingProvider()
    tracker = CostTracker()
    llm = LLM(provider=provider, model="m", cost_tracker=tracker, tracing=False)
    await llm.complete("q1", tag="search")
    await llm.complete("q2", tag="search")
    await llm.complete("q3", tag="summary")
    snap = tracker.snapshot()
    assert abs(snap["by_tag"]["search"] - 0.002) < 1e-9
    assert abs(snap["by_tag"]["summary"] - 0.001) < 1e-9
    assert snap["total_prompt_tokens"] == 30


@pytest.mark.asyncio
async def test_retry_policy_rescues_flaky_provider():
    provider = FailThenSucceedProvider(fail_count=2)
    llm = LLM(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(max_attempts=5, initial_delay=0.001, jitter=0),
        tracing=False,
    )
    result = await llm.complete("hi")
    assert result.content == "ok"
    assert provider.calls == 3
