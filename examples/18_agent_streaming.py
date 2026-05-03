"""Stream agent execution events as they happen.

Run::

    python examples/18_agent_streaming.py
"""

from __future__ import annotations

import asyncio

from actants import LLM, Agent
from actants.agents import (
    AgentRunCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)


async def main() -> None:
    agent = Agent(llm=LLM(model="llama3.2"))

    print("Q: Tell me a one-line joke about Python.")
    print("A: ", end="", flush=True)
    async for event in agent.stream("Tell me a one-line joke about Python."):
        match event:
            case AgentTextDelta(text=t):
                print(t, end="", flush=True)
            case AgentToolCallStarted(call=c):
                print(f"\n  → {c.name}({c.arguments})", flush=True)
            case AgentToolCallCompleted(value=v, ok=ok):
                marker = "✓" if ok else "✗"
                print(f"\n  {marker} {v}", flush=True)
            case AgentRunCompleted(content=final):
                print(f"\n\n[done — {len(final)} chars]")


if __name__ == "__main__":
    asyncio.run(main())
