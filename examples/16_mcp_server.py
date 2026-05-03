"""Expose your tools as an MCP server in two lines.

Once running, point any MCP client (Claude Desktop, IDEs, other agents) at this
process and they get to call your tools.

Run as stdio (the default — Claude Desktop launches stdio servers as subprocesses):
    python examples/16_mcp_server.py

Or run as HTTP for remote clients:
    python examples/16_mcp_server.py --http
"""

from __future__ import annotations

import sys

from actants.mcp import serve
from actants.tools.registry import ToolRegistry


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    async def reverse(text: str) -> str:
        return text[::-1]

    registry.register_function(
        "add",
        "Add two integers and return the sum",
        add,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    registry.register_function(
        "reverse",
        "Reverse a string",
        reverse,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    return registry


if __name__ == "__main__":
    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    serve(build_tools(), transport=transport, name="actants-demo")
