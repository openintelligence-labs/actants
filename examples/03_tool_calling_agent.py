"""An Ollama-powered agent with two tools: a calculator and a clock.

Prereqs: `ollama pull llama3.1` (or another model with tool-call support).

Run: `python examples/03_tool_calling_agent.py`
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from actants import LLM, ToolRegistry


async def add(a: float, b: float) -> float:
    return a + b


async def utc_now() -> str:
    return datetime.now(UTC).isoformat()


async def main() -> None:
    tools = ToolRegistry()
    tools.register_function(
        "add",
        "Add two numbers.",
        add,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    tools.register_function(
        "utc_now",
        "Return the current UTC time in ISO-8601.",
        utc_now,
        input_schema={"type": "object", "properties": {}},
    )

    llm = LLM(model="llama3.1")
    result = await llm.run_agent(
        "What's 17 + 25, and what time is it right now in UTC?",
        tools=tools,
        max_steps=4,
    )
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
