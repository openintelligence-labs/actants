from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from actants.llm.client import LLM
from actants.tools.registry import ToolRegistry


class ScriptedProvider(BaseLLMProvider):
    """Returns a pre-programmed sequence of responses. Records each call's messages."""

    name = "scripted"
    supports_tool_calls = True

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []

    async def complete(
        self,
        messages,
        model,
        temperature=0.7,
        max_tokens=None,
        *,
        tools=None,
        **kwargs,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        return self._responses.pop(0)

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


def _result(content: str, tool_calls: list[ToolCall] | None = None) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="test",
        provider="scripted",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        tool_calls=tool_calls or [],
    )


@pytest.mark.asyncio
async def test_agent_loop_dispatches_tools_then_returns_final_answer():
    provider = ScriptedProvider(
        [
            _result("", [ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            _result("Result is 5"),
        ]
    )
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add two numbers", add)
    llm = LLM(provider=provider, model="test")

    final = await llm.run_agent("What's 2 + 3?", tools=registry, max_steps=3)

    assert final.content == "Result is 5"
    # First call saw tool specs
    assert provider.tools_seen[0] is not None
    assert provider.tools_seen[0][0].name == "add"
    # Second call saw the tool_result message threaded back
    second_msgs = provider.calls[1]
    assert any(m.role == "tool" and m.tool_call_id == "c1" for m in second_msgs)


@pytest.mark.asyncio
async def test_agent_loop_raises_if_exceeds_max_steps():
    provider = ScriptedProvider(
        [
            _result("", [ToolCall(id="c1", name="noop", arguments={})]),
            _result("", [ToolCall(id="c2", name="noop", arguments={})]),
        ]
    )
    registry = ToolRegistry()

    async def noop() -> str:
        return "ok"

    registry.register_function("noop", "does nothing", noop)
    llm = LLM(provider=provider, model="test")

    with pytest.raises(RuntimeError, match="max_steps"):
        await llm.run_agent("go", tools=registry, max_steps=2)


@pytest.mark.asyncio
async def test_agent_loop_handles_tool_error_by_feeding_error_back():
    provider = ScriptedProvider(
        [
            _result("", [ToolCall(id="c1", name="boom", arguments={})]),
            _result("I handled the error"),
        ]
    )
    registry = ToolRegistry()

    async def boom() -> None:
        raise RuntimeError("simulated failure")

    registry.register_function("boom", "fails", boom)
    llm = LLM(provider=provider, model="test")

    final = await llm.run_agent("go", tools=registry, max_steps=3)
    assert final.content == "I handled the error"
    second_msgs = provider.calls[1]
    tool_msg = next(m for m in second_msgs if m.role == "tool")
    assert "simulated failure" in tool_msg.content
