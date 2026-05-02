"""OpenTelemetry GenAI semantic conventions (semconv v1.40.0+).

Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/
Status: Development. We follow current spec; flip default behaviour when GenAI
goes Stable. Set ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` to
get newer experimental attributes; otherwise emit current spec verbatim.

Cost is namespaced under ``actants.cost.*`` because the spec does not define
a cost attribute — we don't squat on ``gen_ai.*``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from opentelemetry import trace

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

_TRACER_NAME = "actants"

# Operation enum from gen_ai-spans.md.
Operation = Literal[
    "chat",
    "generate_content",
    "text_completion",
    "embeddings",
    "create_agent",
    "invoke_agent",
    "execute_tool",
]

# Provider enum from gen_ai attribute registry. Add new ones here as they ship.
KNOWN_PROVIDERS: set[str] = {
    "openai",
    "anthropic",
    "gcp.gen_ai",
    "gcp.vertex_ai",
    "aws.bedrock",
    "az.ai.openai",
    "cohere",
    "mistral_ai",
    "groq",
    "ollama",
    "ibm.watsonx.ai",
    "xai",
    "deepseek",
    "perplexity",
}


def get_tracer() -> Tracer:
    return trace.get_tracer(_TRACER_NAME)


def _experimental_optin() -> bool:
    return "gen_ai_latest_experimental" in os.environ.get(
        "OTEL_SEMCONV_STABILITY_OPT_IN", ""
    ).split(",")


def _set(span: Span, key: str, value: Any) -> None:
    if value is None:
        return
    span.set_attribute(key, value)


@asynccontextmanager
async def chat_span(
    *,
    model: str,
    provider: str,
    conversation_id: str | None = None,
    request_max_tokens: int | None = None,
    request_temperature: float | None = None,
    request_top_p: float | None = None,
    streaming: bool = False,
):
    """Open a span for one LLM chat call.

    Span name: ``"chat <model>"`` per spec. The yielded span has ``record_response()``
    bound on it so callers can attach response-side attributes after the call returns
    without juggling OTel imports themselves.
    """
    name = f"chat {model}"
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        _set(span, "gen_ai.operation.name", "chat")
        _set(span, "gen_ai.provider.name", provider)
        _set(span, "gen_ai.request.model", model)
        _set(span, "gen_ai.conversation.id", conversation_id)
        _set(span, "gen_ai.request.max_tokens", request_max_tokens)
        _set(span, "gen_ai.request.temperature", request_temperature)
        _set(span, "gen_ai.request.top_p", request_top_p)
        _set(span, "gen_ai.request.stream", streaming)
        yield span


def record_response(
    span: Span,
    *,
    response_model: str | None = None,
    response_id: str | None = None,
    finish_reasons: list[str] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    time_to_first_chunk_s: float | None = None,
    cost_usd: float | None = None,
) -> None:
    """Attach response attributes after the LLM call returns.

    Cost is non-spec — namespaced under ``actants.cost.usd`` so a future
    ``gen_ai.cost.*`` won't collide.
    """
    _set(span, "gen_ai.response.model", response_model)
    _set(span, "gen_ai.response.id", response_id)
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
    _set(span, "gen_ai.usage.input_tokens", input_tokens)
    _set(span, "gen_ai.usage.output_tokens", output_tokens)
    _set(span, "gen_ai.usage.cache_read.input_tokens", cache_read_tokens)
    _set(span, "gen_ai.usage.cache_creation.input_tokens", cache_creation_tokens)
    _set(span, "gen_ai.response.time_to_first_chunk", time_to_first_chunk_s)
    _set(span, "actants.cost.usd", cost_usd)


@asynccontextmanager
async def execute_tool_span(
    *,
    tool_name: str,
    tool_call_id: str | None = None,
    tool_description: str | None = None,
):
    """Open a span for one tool execution. Span kind: INTERNAL."""
    from opentelemetry.trace import SpanKind

    name = f"execute_tool {tool_name}"
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
        _set(span, "gen_ai.operation.name", "execute_tool")
        _set(span, "gen_ai.tool.name", tool_name)
        _set(span, "gen_ai.tool.call.id", tool_call_id)
        _set(span, "gen_ai.tool.description", tool_description)
        yield span


@asynccontextmanager
async def invoke_agent_span(
    *,
    agent_name: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
):
    """Open a parent span around a full agent run. Children: chat, execute_tool."""
    name = f"invoke_agent {agent_name}"
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        _set(span, "gen_ai.operation.name", "invoke_agent")
        _set(span, "gen_ai.agent.name", agent_name)
        _set(span, "gen_ai.agent.id", agent_id)
        _set(span, "gen_ai.conversation.id", conversation_id)
        yield span


@asynccontextmanager
async def embeddings_span(
    *,
    model: str,
    provider: str,
    input_count: int | None = None,
):
    """Open a span for an embeddings request."""
    name = f"embeddings {model}"
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        _set(span, "gen_ai.operation.name", "embeddings")
        _set(span, "gen_ai.provider.name", provider)
        _set(span, "gen_ai.request.model", model)
        if input_count is not None:
            _set(span, "actants.embeddings.input_count", input_count)
        yield span
