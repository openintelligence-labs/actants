"""Stateful Agent with conversation memory and lifecycle hooks.

Run::

    pip install -e ".[cli]"
    python examples/10_stateful_agent.py
"""

from __future__ import annotations

import asyncio

from agentic_kit import LLM, Agent, AgentHooks, ConversationMemory, ToolRegistry


async def main() -> None:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function(
        "add",
        "Add two integers",
        add,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )

    async def on_tool_call(call, value):
        print(f"  → tool {call.name}({call.arguments}) = {value}")

    hooks = AgentHooks(on_tool_call=on_tool_call)

    agent = Agent(
        llm=LLM(model="llama3.2"),
        tools=registry,
        memory=ConversationMemory(system="You are a helpful math assistant. Use tools."),
        hooks=hooks,
    )

    r1 = await agent.run("What is 17 + 25?")
    print("Q1:", r1.content)

    r2 = await agent.run("And what is that plus 100?")
    print("Q2:", r2.content)


if __name__ == "__main__":
    asyncio.run(main())
