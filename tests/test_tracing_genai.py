"""OTel GenAI semconv conformance tests."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from actants.llm.base import CompletionResult, TokenUsage
from actants.tracing import genai as genai_mod
from actants.tracing import otel as otel_mod
from actants.tracing.genai import (
    KNOWN_PROVIDERS,
    chat_span,
    embeddings_span,
    execute_tool_span,
    invoke_agent_span,
    record_response,
)
from actants.tracing.otel import instrument_llm


@pytest.fixture
def exporter(monkeypatch):
    """Inject a per-test TracerProvider so spans land in our in-memory exporter."""
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(genai_mod, "get_tracer", lambda: provider.get_tracer("actants"))
    yield exp
    exp.clear()


@pytest.mark.asyncio
async def test_chat_span_uses_spec_name_and_attrs(exporter):
    async with chat_span(
        model="llama3.2",
        provider="ollama",
        conversation_id="conv-1",
        streaming=True,
    ) as span:
        record_response(
            span,
            response_model="llama3.2",
            input_tokens=42,
            output_tokens=15,
            cost_usd=0.0,
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "chat llama3.2"
    attrs = dict(s.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "ollama"
    assert attrs["gen_ai.request.model"] == "llama3.2"
    assert attrs["gen_ai.conversation.id"] == "conv-1"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.usage.input_tokens"] == 42
    assert attrs["gen_ai.usage.output_tokens"] == 15
    assert attrs["actants.cost.usd"] == 0.0


@pytest.mark.asyncio
async def test_execute_tool_span(exporter):
    async with execute_tool_span(
        tool_name="search",
        tool_call_id="call_abc",
        tool_description="search the web",
    ):
        pass
    spans = exporter.get_finished_spans()
    s = spans[0]
    assert s.name == "execute_tool search"
    attrs = dict(s.attributes)
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "search"
    assert attrs["gen_ai.tool.call.id"] == "call_abc"


@pytest.mark.asyncio
async def test_invoke_agent_wraps_chat_span(exporter):
    async with (
        invoke_agent_span(agent_name="researcher", conversation_id="conv-x"),
        chat_span(model="llama3.2", provider="ollama"),
    ):
        pass
    spans = exporter.get_finished_spans()
    # Two spans: child chat, parent invoke_agent — order depends on close order.
    names = sorted(s.name for s in spans)
    assert names == ["chat llama3.2", "invoke_agent researcher"]
    parent = next(s for s in spans if s.name.startswith("invoke_agent"))
    child = next(s for s in spans if s.name.startswith("chat"))
    # Child must reference parent's span context.
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


@pytest.mark.asyncio
async def test_embeddings_span(exporter):
    async with embeddings_span(model="nomic-embed-text", provider="ollama", input_count=3):
        pass
    spans = exporter.get_finished_spans()
    s = spans[0]
    assert s.name == "embeddings nomic-embed-text"
    attrs = dict(s.attributes)
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["actants.embeddings.input_count"] == 3


def test_known_providers_includes_ollama_first_class():
    """Local-first wedge requires Ollama to be a spec'd provider name."""
    assert "ollama" in KNOWN_PROVIDERS


@pytest.mark.asyncio
async def test_record_response_skips_none_attributes(exporter):
    async with chat_span(model="m", provider="ollama") as span:
        record_response(span, input_tokens=10)  # only one attr; rest None

    s = exporter.get_finished_spans()[0]
    attrs = dict(s.attributes)
    assert attrs["gen_ai.usage.input_tokens"] == 10
    # Things we didn't set should not appear.
    assert "gen_ai.response.model" not in attrs
    assert "actants.cost.usd" not in attrs


@pytest.fixture
def otel_exporter(monkeypatch):
    """Same injection as `exporter`, but against `tracing.otel`'s own get_tracer."""
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(otel_mod, "get_tracer", lambda: provider.get_tracer("actants"))
    yield exp
    exp.clear()


@pytest.mark.asyncio
async def test_instrument_llm_records_attributes_from_a_full_result(otel_exporter):
    @instrument_llm
    async def call():
        return CompletionResult(
            content="hi",
            model="llama3.2",
            provider="ollama",
            usage=TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            cost_usd=0.5,
        )

    result = await call()
    assert result.content == "hi"

    attrs = dict(otel_exporter.get_finished_spans()[0].attributes)
    assert attrs["llm.prompt_tokens"] == 7
    assert attrs["llm.completion_tokens"] == 3
    assert attrs["llm.cost_usd"] == 0.5
    assert attrs["llm.model"] == "llama3.2"
    assert attrs["llm.provider"] == "ollama"


@pytest.mark.asyncio
async def test_instrument_llm_survives_a_result_with_usage_but_no_cost(otel_exporter):
    """Regression: the guard checked only `usage`, then read four more attributes.

    `instrument_llm` is public and can wrap any coroutine. A result carrying `usage`
    but no `cost_usd`/`model`/`provider` used to raise AttributeError out of the
    tracing layer, failing a call that had already succeeded.
    """

    class PartialResult:
        usage = TokenUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5)

    @instrument_llm
    async def call():
        return PartialResult()

    result = await call()  # must not raise
    assert isinstance(result, PartialResult)

    attrs = dict(otel_exporter.get_finished_spans()[0].attributes)
    assert attrs["llm.prompt_tokens"] == 4
    assert attrs["llm.completion_tokens"] == 1
    # Absent attributes are simply not recorded.
    assert "llm.cost_usd" not in attrs
    assert "llm.model" not in attrs


@pytest.mark.asyncio
async def test_instrument_llm_passes_through_a_result_with_no_usage(otel_exporter):
    @instrument_llm
    async def call():
        return "plain string"

    assert await call() == "plain string"
    attrs = dict(otel_exporter.get_finished_spans()[0].attributes)
    assert "llm.prompt_tokens" not in attrs


@pytest.mark.asyncio
async def test_instrument_llm_preserves_the_wrapped_name_and_arguments(otel_exporter):
    @instrument_llm
    async def named_call(a, *, b):
        return a + b

    assert named_call.__name__ == "named_call"
    assert await named_call(1, b=2) == 3
    assert otel_exporter.get_finished_spans()[0].name == "named_call"
