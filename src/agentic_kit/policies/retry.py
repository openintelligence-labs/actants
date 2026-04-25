from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff policy with optional jitter.

    ``retry_on`` is a tuple of exception types that trigger a retry. Any other exception
    propagates immediately.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: float = 0.1
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def delay_for(self, attempt: int) -> float:
        """Delay before attempt N (1-indexed). attempt=1 means no delay before the first call."""
        if attempt <= 1:
            return 0.0
        raw = self.initial_delay * (self.backoff_factor ** (attempt - 2))
        raw = min(raw, self.max_delay)
        if self.jitter > 0:
            raw *= 1 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, raw)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
) -> T:
    """Run ``func`` with retries per the given policy.

    Re-raises the final exception if all attempts fail. ``func`` must be a zero-arg coroutine;
    bind arguments via ``functools.partial`` or a lambda.
    """
    policy = policy or RetryPolicy()
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        delay = policy.delay_for(attempt)
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            return await func()
        except policy.retry_on as exc:
            last_exc = exc
            log.warning(
                "retry_attempt_failed",
                attempt=attempt,
                max_attempts=policy.max_attempts,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if attempt >= policy.max_attempts:
                raise
    assert last_exc is not None  # pragma: no cover
    raise last_exc
