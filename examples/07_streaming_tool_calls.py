"""Stream a full agent loop — watch text deltas and tool calls as they happen.

Prereqs: `ollama pull llama3.1` (needs tool-call-capable model).

Run: `python examples/07_streaming_tool_calls.py`
"""

from __future__ import annotations

import asyncio
import sys

from agentic_kit import LLM, ToolRegistry
from agentic_kit.llm.base import FinishDelta, TextDelta, ToolCallDelta, UsageDelta


async def multiply(a: float, b: float) -> float:
    return a * b


async def main() -> None:
    tools = ToolRegistry()
    tools.register_function(
        "multiply",
        "Multiply two numbers.",
        multiply,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )

    llm = LLM(model="llama3.1")
    async for event in llm.run_agent_stream(
        "What is 234 * 17? Think step by step and use the multiply tool.",
        tools=tools,
    ):
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, ToolCallDelta):
            print(
                f"\n[tool call] {event.tool_call.name}({event.tool_call.arguments})",
                flush=True,
            )
        elif isinstance(event, UsageDelta):
            print(f"\n[usage] {event.usage.total_tokens} tokens")
        elif isinstance(event, FinishDelta):
            pass
    print()


if __name__ == "__main__":
    asyncio.run(main())
