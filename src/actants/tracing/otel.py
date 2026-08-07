from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

_TRACER_NAME = "actants"

_R = TypeVar("_R")


def get_tracer() -> Tracer:
    return trace.get_tracer(_TRACER_NAME)


@asynccontextmanager
async def llm_span(name: str, **attrs: Any) -> AsyncIterator[Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"actants.{k}", v)
        yield span


def instrument_llm(func: Callable[..., Awaitable[_R]]) -> Callable[..., Awaitable[_R]]:
    """Decorator to wrap LLM calls in a span with usage attributes."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> _R:
        async with llm_span(func.__name__) as span:
            result = await func(*args, **kwargs)
            # Each attribute is probed independently. This decorator is public and can
            # wrap any coroutine, so a result that carries `usage` but not `cost_usd`
            # is an ordinary duck-typing miss, not a bug -- and a tracing wrapper must
            # never be the thing that raises out of an otherwise successful call.
            usage = getattr(result, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                if prompt_tokens is not None:
                    span.set_attribute("llm.prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if completion_tokens is not None:
                    span.set_attribute("llm.completion_tokens", completion_tokens)
            for attr, key in (
                ("cost_usd", "llm.cost_usd"),
                ("model", "llm.model"),
                ("provider", "llm.provider"),
            ):
                value = getattr(result, attr, None)
                if value is not None:
                    span.set_attribute(key, value)
            return result

    return wrapper
