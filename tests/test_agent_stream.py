"""Agent.stream() tests — streaming-first event surface."""

from __future__ import annotations

import pytest

from agentic_kit.agents import (
    Agent,
    AgentRunCompleted,
    AgentStepCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)
from agentic_kit.llm.client import LLM
from agentic_kit.testing import (
    FakeLLMProvider,
    fake_completion,
    fake_tool_call_completion,
)
from agentic_kit.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_stream_yields_text_deltas_then_run_completed():
    provider = FakeLLMProvider([fake_completion("Hello!")])
    agent = Agent(llm=LLM(provider=provider, model="fake"))

    events = [e async for e in agent.stream("hi")]

    text_chunks = [e for e in events if isinstance(e, AgentTextDelta)]
    final = [e for e in events if isinstance(e, AgentRunCompleted)]
    step_done = [e for e in events if isinstance(e, AgentStepCompleted)]

    assert len(text_chunks) == len("Hello!")  # FakeLLM streams char-by-char
    assert "".join(c.text for c in text_chunks) == "Hello!"
    assert len(step_done) == 1
    assert len(final) == 1
    assert final[0].content == "Hello!"
    # Last event must be AgentRunCompleted.
    assert isinstance(events[-1], AgentRunCompleted)


@pytest.mark.asyncio
async def test_stream_emits_tool_call_lifecycle():
    provider = FakeLLMProvider(
        [
            fake_tool_call_completion("add", {"a": 2, "b": 3}, call_id="t1"),
            fake_completion("Result is 5"),
        ]
    )
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add", add)

    agent = Agent(llm=LLM(provider=provider, model="fake"), tools=registry)
    events = [e async for e in agent.stream("2 + 3?")]

    started = [e for e in events if isinstance(e, AgentToolCallStarted)]
    completed = [e for e in events if isinstance(e, AgentToolCallCompleted)]
    final = [e for e in events if isinstance(e, AgentRunCompleted)]

    assert len(started) == 1 and started[0].call.name == "add"
    assert len(completed) == 1 and completed[0].ok is True
    assert completed[0].value == 5
    assert final[0].content == "Result is 5"


@pytest.mark.asyncio
async def test_stream_updates_memory_to_match_run():
    """After streaming completes, conversation memory should match what run() produces."""
    provider = FakeLLMProvider([fake_completion("Hi")])
    agent = Agent(llm=LLM(provider=provider, model="fake"))

    [_ async for _ in agent.stream("hello")]

    msgs = agent.memory.messages()
    roles = [m.role for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[1].content == "Hi"


@pytest.mark.asyncio
async def test_stream_propagates_errors_via_on_error_hook():
    from agentic_kit.agents import AgentHooks

    captured = []

    async def on_err(exc):
        captured.append(exc)

    # No tool registry but model asks for one → RuntimeError
    provider = FakeLLMProvider([fake_tool_call_completion("missing", {})])
    agent = Agent(
        llm=LLM(provider=provider, model="fake"),
        hooks=AgentHooks(on_error=on_err),
    )
    with pytest.raises(RuntimeError):
        async for _ in agent.stream("trigger error"):
            pass
    assert len(captured) == 1
