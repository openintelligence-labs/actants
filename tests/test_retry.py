from __future__ import annotations

import pytest

from agentic_kit.policies.retry import RetryPolicy, retry_async


@pytest.mark.asyncio
async def test_retry_succeeds_eventually():
    attempts = {"n": 0}

    async def flaky() -> int:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return 42

    result = await retry_async(flaky, RetryPolicy(max_attempts=5, initial_delay=0.001, jitter=0))
    assert result == 42
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    async def always_fails() -> None:
        calls["n"] += 1
        raise ValueError("always")

    with pytest.raises(ValueError):
        await retry_async(always_fails, RetryPolicy(max_attempts=3, initial_delay=0.001, jitter=0))
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_only_on_configured_exceptions():
    async def boom() -> None:
        raise KeyError("nope")

    # KeyError is not in retry_on, so it propagates on first failure
    with pytest.raises(KeyError):
        await retry_async(
            boom,
            RetryPolicy(max_attempts=5, initial_delay=0.001, jitter=0, retry_on=(ValueError,)),
        )


def test_retry_delay_schedule_is_exponential():
    p = RetryPolicy(max_attempts=5, initial_delay=1.0, backoff_factor=2.0, jitter=0, max_delay=100)
    assert p.delay_for(1) == 0.0
    assert p.delay_for(2) == 1.0
    assert p.delay_for(3) == 2.0
    assert p.delay_for(4) == 4.0
    assert p.delay_for(5) == 8.0


def test_retry_delay_clamped_by_max():
    p = RetryPolicy(max_attempts=10, initial_delay=1.0, backoff_factor=10.0, jitter=0, max_delay=5)
    assert p.delay_for(5) == 5.0
