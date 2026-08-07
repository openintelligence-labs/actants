"""Regression tests for runtime-correctness defects found by adversarial review.

Each test here failed before its corresponding fix. They cover the paths where a bug
is silent — wrong cost, a cache collision, a swallowed tool call — rather than loud.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator

import pytest

from actants.agents.agent import Agent
from actants.cache.memory import InMemoryCache, make_key
from actants.cost.tracker import CostTracker
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.client import LLM
from actants.policies.fallback import FallbackProvider
from actants.tools.base import ToolResult, serialize_tool_result
from actants.tools.registry import ToolRegistry


class RecordingProvider(BaseLLMProvider):
    """Counts completions so cache hits are observable."""

    name = "recording"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, name: str = "recording") -> None:
        self.name = name
        self.completions = 0

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> CompletionResult:
        self.completions += 1
        return CompletionResult(content="ok", model=model, provider=self.name, usage=TokenUsage())

    async def health(self) -> bool:
        return True


# --------------------------------------------------------------------------------------
# Cold import: the whole `cost` subpackage was unimportable via a circular import.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "from actants import CostTracker",
        "from actants import PRICING",
        "from actants import estimate_cost",
        "import actants.cost",
        "import actants.cost.tracker",
        "import actants.cost.pricing",
    ],
)
def test_public_symbols_import_on_a_cold_interpreter(statement: str) -> None:
    """Importing cost before anything else must not hit a circular import."""
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"`{statement}` failed on a cold interpreter:\n{result.stderr}"


# --------------------------------------------------------------------------------------
# Cache keys must cover everything that changes the answer.
# --------------------------------------------------------------------------------------


def test_make_key_distinguishes_max_tokens() -> None:
    msgs = [ChatMessage(role="user", content="hi")]
    assert make_key(msgs, "m", 0.5, max_tokens=16) != make_key(msgs, "m", 0.5, max_tokens=4096)


def test_make_key_distinguishes_provider() -> None:
    msgs = [ChatMessage(role="user", content="hi")]
    assert make_key(msgs, "m", 0.5, provider="ollama") != make_key(
        msgs, "m", 0.5, provider="openai"
    )


def test_make_key_distinguishes_tools() -> None:
    msgs = [ChatMessage(role="user", content="hi")]
    spec = ToolSpec(name="a", description="d")
    assert make_key(msgs, "m", 0.5) != make_key(msgs, "m", 0.5, tools=[spec])


def test_make_key_distinguishes_tool_call_structure() -> None:
    """Two messages differing only in tool-call payload must not share a key."""
    a = [ChatMessage(role="assistant", tool_calls=[ToolCall(id="1", name="x", arguments={})])]
    b = [ChatMessage(role="assistant", tool_calls=[ToolCall(id="1", name="y", arguments={})])]
    assert make_key(a, "m", 0.5) != make_key(b, "m", 0.5)


def test_make_key_content_cannot_forge_a_field_boundary() -> None:
    """Message content must not be able to impersonate the role/content separators."""
    two = [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")]
    one = [ChatMessage(role="user", content="a\x01user\x00b")]
    assert make_key(two, "m", 0.5) != make_key(one, "m", 0.5)


def test_make_key_distinguishes_small_temperature_differences() -> None:
    msgs = [ChatMessage(role="user", content="hi")]
    assert make_key(msgs, "m", 0.0001) != make_key(msgs, "m", 0.0002)


async def test_complete_does_not_serve_cached_result_across_max_tokens() -> None:
    provider = RecordingProvider()
    llm = LLM(provider=provider, model="m", cache=InMemoryCache(), tracing=False)
    await llm.complete("hi", max_tokens=8)
    await llm.complete("hi", max_tokens=4096)
    assert provider.completions == 2


async def test_complete_does_not_serve_cached_result_across_providers() -> None:
    """Two LLMs sharing one cache must not read each other's answers."""
    cache = InMemoryCache()
    first = RecordingProvider(name="ollama")
    second = RecordingProvider(name="openai")
    await LLM(provider=first, model="m", cache=cache, tracing=False).complete("hi")
    await LLM(provider=second, model="m", cache=cache, tracing=False).complete("hi")
    assert second.completions == 1


# --------------------------------------------------------------------------------------
# Tool dispatch must survive model-controlled input.
# --------------------------------------------------------------------------------------


async def test_unknown_tool_name_is_reported_not_raised() -> None:
    """A hallucinated tool name is normal model behaviour, not a crash."""
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add two integers", add)
    result = await registry.call("does_not_exist")
    assert result.ok is False
    assert "does_not_exist" in (result.error or "")


async def test_bad_tool_arguments_are_reported_not_raised() -> None:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add two integers", add)
    result = await registry.call("add", nonexistent=1)
    assert result.ok is False


def test_serialize_tool_result_survives_a_circular_value() -> None:
    cyclic: dict = {}
    cyclic["self"] = cyclic
    payload = serialize_tool_result(ToolResult(ok=True, value=cyclic))
    assert "could not be serialized" in payload


def test_serialize_tool_result_survives_a_value_whose_str_raises() -> None:
    class Explosive:
        def __str__(self) -> str:
            raise ValueError("boom")

        __repr__ = __str__

    payload = serialize_tool_result(ToolResult(ok=True, value=Explosive()))
    assert "could not be serialized" in payload


async def test_agent_loop_survives_an_unserializable_tool_return() -> None:
    """One badly-behaved tool must not abort the whole run."""

    class ToolThenAnswer(BaseLLMProvider):
        name = "two-step"
        supports_tool_calls = True

        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
        ) -> CompletionResult:
            self.calls += 1
            if self.calls == 1:
                return CompletionResult(
                    content="",
                    model=model,
                    provider=self.name,
                    usage=TokenUsage(),
                    tool_calls=[ToolCall(id="1", name="cyclic", arguments={})],
                )
            return CompletionResult(
                content="done", model=model, provider=self.name, usage=TokenUsage()
            )

        async def health(self) -> bool:
            return True

    registry = ToolRegistry()

    async def cyclic() -> dict:
        out: dict = {}
        out["self"] = out
        return out

    registry.register_function("cyclic", "Returns a cyclic structure", cyclic)
    llm = LLM(provider=ToolThenAnswer(), model="m", tracing=False)
    result = await llm.run_agent("go", registry)
    assert result.content == "done"


# --------------------------------------------------------------------------------------
# Streamed runs must report real cost.
# --------------------------------------------------------------------------------------


class CostedStreamProvider(BaseLLMProvider):
    """Emits a UsageDelta carrying a non-zero cost, like every paid provider does."""

    name = "costed"
    supports_tool_calls = True
    supports_streaming_tools = True

    async def complete(self, messages, model, **kwargs) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="hi")
        yield UsageDelta(
            usage=TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
            cost_usd=1.25,
        )
        yield FinishDelta(reason="stop")


async def test_agent_stream_records_real_cost() -> None:
    """Agent.stream() dropped UsageDelta.cost_usd and reported every run as free."""
    tracker = CostTracker()
    llm = LLM(provider=CostedStreamProvider(), model="m", tracing=False, cost_tracker=tracker)
    agent = Agent(llm=llm)
    async for _ in agent.stream("hi"):
        pass
    assert tracker.snapshot()["total_usd"] == pytest.approx(1.25)


async def test_agent_stream_step_completion_carries_cost() -> None:
    llm = LLM(provider=CostedStreamProvider(), model="m", tracing=False)
    agent = Agent(llm=llm)
    costs = [
        event.completion.cost_usd
        async for event in agent.stream("hi")
        if type(event).__name__ == "AgentStepCompleted"
    ]
    assert costs == [pytest.approx(1.25)]


# --------------------------------------------------------------------------------------
# Fallback must not corrupt or downgrade a stream.
# --------------------------------------------------------------------------------------


class PartialThenFailProvider(BaseLLMProvider):
    name = "partial"

    async def complete(self, messages, model, **kwargs) -> CompletionResult:
        raise NotImplementedError

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="Hello ")
        yield TextDelta(text="wor")
        raise RuntimeError("connection reset")

    async def health(self) -> bool:
        return True


class HealthyProvider(BaseLLMProvider):
    name = "healthy"
    supports_tool_calls = True
    supports_streaming_tools = True

    async def complete(self, messages, model, **kwargs) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="Hello world")
        yield ToolCallDelta(tool_call=ToolCall(id="1", name="t", arguments={"a": 1}))
        yield FinishDelta(reason="tool_calls")


async def test_fallback_stream_does_not_replay_already_emitted_text() -> None:
    """Failing over mid-stream spliced two completions into one response."""
    chain = FallbackProvider([(PartialThenFailProvider(), None), (HealthyProvider(), None)])
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="connection reset"):
        async for chunk in chain.stream([ChatMessage(role="user", content="hi")], "m"):
            seen.append(chunk)
    assert seen == ["Hello ", "wor"]


async def test_fallback_stream_still_fails_over_before_emitting() -> None:
    class DeadProvider(BaseLLMProvider):
        name = "dead"

        async def complete(self, messages, model, **kwargs) -> CompletionResult:
            raise NotImplementedError

        async def stream_events(
            self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
        ) -> AsyncIterator[StreamEvent]:
            raise RuntimeError("connection refused")
            yield  # unreachable; keeps this an async generator

        async def health(self) -> bool:
            return False

    chain = FallbackProvider([(DeadProvider(), None), (HealthyProvider(), None)])
    chunks = [c async for c in chain.stream([ChatMessage(role="user", content="hi")], "m")]
    assert chunks == ["Hello world"]


async def test_fallback_preserves_tool_call_events() -> None:
    """The inherited stream_events wrapped `stream`, silently dropping tool calls."""
    chain = FallbackProvider([(HealthyProvider(), None)])
    events = [
        e
        async for e in chain.stream_events(
            messages=[ChatMessage(role="user", content="hi")], model="m", tools=[]
        )
    ]
    assert any(isinstance(e, ToolCallDelta) for e in events)


def test_fallback_reports_tool_support_of_the_weakest_link() -> None:
    class NoTools(BaseLLMProvider):
        name = "no-tools"
        supports_tool_calls = False

        async def complete(self, messages, model, **kwargs) -> CompletionResult:
            raise NotImplementedError

        async def health(self) -> bool:
            return True

    assert FallbackProvider([(HealthyProvider(), None)]).supports_tool_calls is True
    assert (
        FallbackProvider([(HealthyProvider(), None), (NoTools(), None)]).supports_tool_calls
        is False
    )
