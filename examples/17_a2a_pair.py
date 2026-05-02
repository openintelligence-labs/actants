"""Two agents in two processes, one calls the other over A2A.

Terminal 1 — the math expert::

    pip install -e ".[a2a]"
    python examples/17_a2a_pair.py serve

Terminal 2 — call it::

    python examples/17_a2a_pair.py call

Or run locally with the in-process Starlette TestClient — see tests/test_a2a_server.py.
"""

from __future__ import annotations

import asyncio
import sys

from agentic_kit import LLM, Agent
from agentic_kit.a2a import RemoteAgent, serve
from agentic_kit.tools.registry import ToolRegistry


def build_math_agent() -> Agent:
    registry = ToolRegistry()

    async def square(n: int) -> int:
        return n * n

    registry.register_function(
        "square",
        "Square an integer",
        square,
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    return Agent(llm=LLM(model="llama3.2"), tools=registry)


async def call_remote() -> None:
    remote = RemoteAgent("http://127.0.0.1:9000", name="math_expert")
    caller = Agent(llm=LLM(model="llama3.2"), tools=ToolRegistry())
    caller.tools.register(remote)
    result = await caller.run("Ask the math_expert what 9 squared is.")
    print("Reply:", result.content)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "serve":
        serve(build_math_agent(), name="math-expert", port=9000)
    elif sys.argv[1] == "call":
        asyncio.run(call_remote())
    else:
        print(f"Usage: {sys.argv[0]} [serve|call]")
        sys.exit(2)


if __name__ == "__main__":
    main()
