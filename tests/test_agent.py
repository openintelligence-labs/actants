from __future__ import annotations

import pytest

from actants.agents import Agent, AgentHooks, AgentResult, ConversationMemory
from actants.llm.client import LLM
from actants.testing import (
    FakeLLMProvider,
    fake_completion,
    fake_tool_call_completion,
)
from actants.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_returns_final_answer_with_no_tools():
    provider = FakeLLMProvider([fake_completion("Hello there!")])
    agent = Agent(llm=LLM(provider=provider, model="fake"))

    result = await agent.run("Hi")

    assert result.content == "Hello there!"
    assert len(result.steps) == 1
    assert result.steps[0].tool_calls == []
    assert len(agent.memory) == 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_dispatches_tool_then_finalizes():
    provider = FakeLLMProvider(
        [
            fake_tool_call_completion("add", {"a": 2, "b": 3}, call_id="t1"),
            fake_completion("Result is 5"),
        ]
    )
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add two ints", add)

    agent = Agent(llm=LLM(provider=provider, model="fake"), tools=registry)
    result = await agent.run("What is 2 + 3?")

    assert result.content == "Result is 5"
    assert len(result.steps) == 2
    assert result.steps[0].tool_calls[0].name == "add"
    assert result.steps[0].tool_results == ["5"]


@pytest.mark.asyncio
async def test_agent_run_returns_an_agent_result_after_dispatching_tools():
    """Regression: the tool loop reused the name holding this run's AgentResult.

    `run()` accumulated its return value in `result`, and the tool-dispatch loop
    assigned each `ToolResult` to that same name. The `assert result is not None`
    before the return was documented as "the loop either sets it or raises", but
    the tool loop also set it -- with the wrong type. Nothing escaped today only
    because the `for/else` raises on exhaustion, so the last write always came
    from the final-answer branch. Any early exit added to that loop would have
    returned a ToolResult from a function annotated `-> AgentResult`.
    """
    provider = FakeLLMProvider(
        [
            fake_tool_call_completion("noop", {}, call_id="t1"),
            fake_completion("done"),
        ]
    )
    registry = ToolRegistry()

    async def noop() -> str:
        return "ok"

    registry.register_function("noop", "Does nothing", noop)

    agent = Agent(llm=LLM(provider=provider, model="fake"), tools=registry)
    result = await agent.run("go")

    assert isinstance(result, AgentResult)
    # A ToolResult has `.ok`/`.value` and no `.steps`; assert the AgentResult shape.
    assert result.content == "done"
    assert len(result.steps) == 2
    assert not hasattr(result, "ok")


@pytest.mark.asyncio
async def test_agent_remembers_across_turns():
    provider = FakeLLMProvider([fake_completion("My name is Bot"), fake_completion("It's Bot")])
    agent = Agent(
        llm=LLM(provider=provider, model="fake"),
        system="You are an assistant",
    )

    await agent.run("What is your name?")
    await agent.run("Say it again?")

    # Second LLM call should have seen the prior assistant turn
    second_call_msgs = provider.calls[1]
    roles = [m.role for m in second_call_msgs]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_agent_hooks_fire_in_order():
    provider = FakeLLMProvider([fake_completion("done")])
    fired: list[str] = []

    async def before(i, msgs):
        fired.append(f"before:{i}")

    async def after(i, completion):
        fired.append(f"after:{i}")

    hooks = AgentHooks(before_step=before, after_step=after)
    agent = Agent(llm=LLM(provider=provider, model="fake"), hooks=hooks)

    await agent.run("hi")

    assert fired == ["before:0", "after:0"]


@pytest.mark.asyncio
async def test_agent_max_steps_raises_when_exceeded():
    # Always returns a tool call → never finalizes
    provider = FakeLLMProvider(
        [
            fake_tool_call_completion("noop", {}),
            fake_tool_call_completion("noop", {}),
        ]
    )
    registry = ToolRegistry()

    async def noop() -> str:
        return "ok"

    registry.register_function("noop", "no-op", noop)
    agent = Agent(llm=LLM(provider=provider, model="fake"), tools=registry, max_steps=2)

    with pytest.raises(RuntimeError, match="max_steps"):
        await agent.run("loop forever")


def test_conversation_memory_trims_to_max_messages():
    mem = ConversationMemory(system="sys", max_messages=4)
    for i in range(10):
        mem.add_user(f"u{i}")
    msgs = mem.messages()
    # System always preserved + 3 most recent user messages = 4 total
    assert msgs[0].role == "system"
    assert len(msgs) == 4
    assert msgs[-1].content == "u9"


def test_conversation_memory_reset_keeps_system():
    mem = ConversationMemory(system="sys")
    mem.add_user("hi")
    mem.add_assistant("hello")
    mem.reset()
    msgs = mem.messages()
    assert len(msgs) == 1
    assert msgs[0].role == "system"
