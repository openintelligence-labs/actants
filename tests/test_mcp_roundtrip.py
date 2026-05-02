"""End-to-end: agentic-kit serves an MCP server, agentic-kit consumes it.

The full-circle test that proves both halves of the MCP module work together.
"""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from agentic_kit.mcp import build_server  # noqa: E402
from agentic_kit.mcp.adapters import mcp_tool_to_agentic_tool  # noqa: E402
from agentic_kit.tools.registry import ToolRegistry  # noqa: E402


@pytest.mark.asyncio
async def test_full_circle_serve_then_consume():
    """Serve a registry, then consume it back as agentic-kit tools.

    What the agent sees on the consumer side must be functionally identical
    to what the producer registered.
    """
    producer = ToolRegistry()

    async def square(n: int) -> int:
        return n * n

    producer.register_function(
        "square",
        "Square an integer",
        square,
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )

    server = build_server(producer, name="math-srv")

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        consumer = ToolRegistry()
        for mcp_tool in listing.tools:
            consumer.register(mcp_tool_to_agentic_tool(mcp_tool, session, server_name="remote"))

        # Consumer-side tool name is prefixed; underlying call still works.
        names = [t.name for t in consumer.list()]
        assert "remote__square" in names

        result = await consumer.call("remote__square", n=9)
        assert result.ok
        # MCP wraps single-int returns in a structured-output envelope; either form is OK.
        assert "81" in str(result.value)
