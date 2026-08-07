"""Cost attribution must survive the switch from ``complete()`` to streaming.

Before 1.0, ``tag`` existed on ``LLM.complete``, ``Agent.run``, and ``Agent.stream``, but
not on ``LLM.stream``, ``LLM.stream_events``, or ``LLM.extract``. A user who tagged their
completions for per-feature cost attribution and then switched one call to streaming lost
that attribution silently: the spend still landed in ``total_usd``, so nothing looked
broken, but ``by_tag`` no longer added up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from actants.agents.agent import Agent
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
from actants.tools.registry import ToolRegistry

COST = 1.25
USAGE = TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110)


class TaggedStreamProvider(BaseLLMProvider):
    """Streams one text delta and a priced UsageDelta, like every paid provider."""

    name = "tagged"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, text: str = "hi") -> None:
        self.text = text

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        return CompletionResult(
            content=self.text,
            model=model,
            provider=self.name,
            usage=USAGE,
            cost_usd=COST,
        )

    async def health(self) -> bool:
        return True

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text=self.text)
        yield UsageDelta(usage=USAGE, cost_usd=COST)
        yield FinishDelta(reason="stop")


def _llm(provider: BaseLLMProvider | None = None) -> tuple[LLM, CostTracker]:
    tracker = CostTracker()
    llm = LLM(
        provider=provider or TaggedStreamProvider(),
        model="m",
        tracing=False,
        cost_tracker=tracker,
    )
    return llm, tracker


# --------------------------------------------------------------------------------------
# LLM.stream
# --------------------------------------------------------------------------------------


async def test_stream_attributes_cost_to_tag() -> None:
    llm, tracker = _llm()
    async for _ in llm.stream("hi", tag="feature-a"):
        pass
    snap = tracker.snapshot()
    assert snap["by_tag"] == {"feature-a": pytest.approx(COST)}
    assert snap["total_usd"] == pytest.approx(COST)


async def test_stream_without_tag_still_records_total() -> None:
    """An untagged stream must still count toward the total, just not under a tag."""
    llm, tracker = _llm()
    async for _ in llm.stream("hi"):
        pass
    snap = tracker.snapshot()
    assert snap["by_tag"] == {}
    assert snap["total_usd"] == pytest.approx(COST)


# --------------------------------------------------------------------------------------
# LLM.stream_events
# --------------------------------------------------------------------------------------


async def test_stream_events_attributes_cost_to_tag() -> None:
    llm, tracker = _llm()
    async for _ in llm.stream_events("hi", tag="feature-b"):
        pass
    assert tracker.snapshot()["by_tag"] == {"feature-b": pytest.approx(COST)}


async def test_stream_events_records_tokens_under_tag_run() -> None:
    """Token counts follow the same path as cost, so they must land too."""
    llm, tracker = _llm()
    async for _ in llm.stream_events("hi", tag="feature-b"):
        pass
    snap = tracker.snapshot()
    assert snap["total_prompt_tokens"] == USAGE.prompt_tokens
    assert snap["total_completion_tokens"] == USAGE.completion_tokens


async def test_streamed_and_completed_spend_share_one_tag() -> None:
    """The whole point: mixing complete() and stream() under one tag sums correctly."""
    llm, tracker = _llm()
    await llm.complete("hi", tag="shared")
    async for _ in llm.stream("hi", tag="shared"):
        pass
    async for _ in llm.stream_events("hi", tag="shared"):
        pass
    assert tracker.snapshot()["by_tag"] == {"shared": pytest.approx(COST * 3)}


# --------------------------------------------------------------------------------------
# LLM.extract
# --------------------------------------------------------------------------------------


class Person(BaseModel):
    name: str


class JSONProvider(TaggedStreamProvider):
    """Returns valid JSON for Person."""

    def __init__(self) -> None:
        super().__init__(text='{"name": "Ada"}')


async def test_extract_attributes_cost_to_tag() -> None:
    llm, tracker = _llm(JSONProvider())
    person = await llm.extract("who?", Person, tag="extraction")
    assert person.name == "Ada"
    assert tracker.snapshot()["by_tag"] == {"extraction": pytest.approx(COST)}


class RepairingProvider(BaseLLMProvider):
    """Returns unparseable output first, valid JSON on the repair attempt."""

    name = "repairing"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        self.calls += 1
        content = "not json at all" if self.calls == 1 else '{"name": "Ada"}'
        return CompletionResult(
            content=content, model=model, provider=self.name, usage=USAGE, cost_usd=COST
        )

    async def health(self) -> bool:
        return True


async def test_extract_tags_every_attempt_including_failed_repairs() -> None:
    """A repair costs real tokens; charging only the successful attempt understates it."""
    provider = RepairingProvider()
    llm, tracker = _llm(provider)
    person = await llm.extract("who?", Person, tag="extraction", max_repairs=1)
    assert person.name == "Ada"
    assert provider.calls == 2
    assert tracker.snapshot()["by_tag"] == {"extraction": pytest.approx(COST * 2)}


async def test_extract_stream_attributes_cost_to_tag() -> None:
    llm, tracker = _llm(JSONProvider())
    seen = [p async for p in llm.extract_stream("who?", Person, tag="extraction")]
    assert seen[-1].name == "Ada"
    assert tracker.snapshot()["by_tag"] == {"extraction": pytest.approx(COST)}


# --------------------------------------------------------------------------------------
# max_repairs semantics
# --------------------------------------------------------------------------------------


async def test_max_repairs_zero_makes_exactly_one_request() -> None:
    """``max_repairs`` counts repairs, not total attempts: 0 means no self-correction."""
    provider = RepairingProvider()
    llm, _ = _llm(provider)
    with pytest.raises(ValueError, match="Failed to extract"):
        await llm.extract("who?", Person, max_repairs=0)
    assert provider.calls == 1


async def test_max_repairs_one_makes_at_most_two_requests() -> None:
    provider = RepairingProvider()
    llm, _ = _llm(provider)
    await llm.extract("who?", Person, max_repairs=1)
    assert provider.calls == 2


# --------------------------------------------------------------------------------------
# Agent paths must not double-count
# --------------------------------------------------------------------------------------


async def test_agent_stream_records_cost_exactly_once() -> None:
    """Agent.stream and LLM.stream_events must not both record the same UsageDelta."""
    llm, tracker = _llm()
    agent = Agent(llm=llm)
    async for _ in agent.stream("hi", tag="agent-tag"):
        pass
    snap = tracker.snapshot()
    assert snap["total_usd"] == pytest.approx(COST)
    assert snap["by_tag"] == {"agent-tag": pytest.approx(COST)}


async def test_agent_run_and_stream_agree_on_tagged_total() -> None:
    llm_a, tracker_a = _llm()
    await Agent(llm=llm_a).run("hi", tag="t")

    llm_b, tracker_b = _llm()
    async for _ in Agent(llm=llm_b).stream("hi", tag="t"):
        pass

    assert tracker_a.snapshot()["by_tag"] == tracker_b.snapshot()["by_tag"]


# --------------------------------------------------------------------------------------
# run_agent_stream
# --------------------------------------------------------------------------------------


class ToolThenAnswerStream(BaseLLMProvider):
    """Streams a tool call on the first pass, then a final answer."""

    name = "tool-stream"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self.calls == 1:
            yield ToolCallDelta(tool_call=ToolCall(id="1", name="ping", arguments={}))
        else:
            yield TextDelta(text="done")
        yield UsageDelta(usage=USAGE, cost_usd=COST)
        yield FinishDelta(reason="stop")


async def test_run_agent_stream_attributes_every_step_to_tag() -> None:
    async def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register_function("ping", "Returns pong", ping)
    llm, tracker = _llm(ToolThenAnswerStream())
    async for _ in llm.run_agent_stream("go", registry, tag="loop"):
        pass
    # Two LLM steps: the tool call and the final answer.
    assert tracker.snapshot()["by_tag"] == {"loop": pytest.approx(COST * 2)}


# --------------------------------------------------------------------------------------
# Unpriced models must still be reported through the streaming path
# --------------------------------------------------------------------------------------


class UnpricedStreamProvider(TaggedStreamProvider):
    name = "definitely-not-a-real-provider"


async def test_streamed_unpriced_model_is_reported_as_untracked() -> None:
    """The honest-unknown-cost reporting must work for streams, not just completions."""
    llm, tracker = _llm(UnpricedStreamProvider())
    async for _ in llm.stream("hi", tag="t"):
        pass
    snap = tracker.snapshot()
    assert snap["untracked_models"] == ["definitely-not-a-real-provider/m"]
    assert tracker.has_untracked_cost
