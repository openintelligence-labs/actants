from __future__ import annotations

from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

_TRACER_NAME = "agentic_kit"


def get_tracer() -> Tracer:
    return trace.get_tracer(_TRACER_NAME)


@asynccontextmanager
async def llm_span(name: str, **attrs: Any):
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"agentic_kit.{k}", v)
        yield span


def instrument_llm(func):
    """Decorator to wrap LLM calls in a span with usage attributes."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with llm_span(func.__name__) as span:
            result = await func(*args, **kwargs)
            if hasattr(result, "usage"):
                span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                span.set_attribute("llm.cost_usd", result.cost_usd)
                span.set_attribute("llm.model", result.model)
                span.set_attribute("llm.provider", result.provider)
            return result

    return wrapper
