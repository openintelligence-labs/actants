"""MCP client tests using the SDK's in-memory transport (no subprocess)."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from actants.mcp.adapters import (  # noqa: E402
    _flatten_content,
    call_result_to_tool_result,
    mcp_tool_to_agentic_tool,
)
from actants.tools.base import ToolError  # noqa: E402


def _build_server() -> FastMCP:
    server = FastMCP("test-server")

    @server.tool()
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    async def fail() -> str:
        """Always raises."""
        raise RuntimeError("boom")

    @server.tool()
    async def echo(text: str) -> str:
        """Echo the text back."""
        return text

    return server


@pytest.mark.asyncio
async def test_list_tools_roundtrip():
    server = _build_server()
    async with create_connected_server_and_client_session(
        server._mcp_server,
    ) as session:
        listing = await session.list_tools()
        names = sorted(t.name for t in listing.tools)
        assert names == ["add", "echo", "fail"]


@pytest.mark.asyncio
async def test_call_tool_via_adapter_returns_value():
    server = _build_server()
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        add_tool = next(t for t in listing.tools if t.name == "add")
        agentic_tool = mcp_tool_to_agentic_tool(add_tool, session, server_name="srv")

        assert agentic_tool.name == "srv__add"
        assert "Add two integers" in agentic_tool.description

        result = await agentic_tool.handler(a=2, b=3)
        # FastMCP returns structured output for typed returns; we accept either.
        if isinstance(result, dict):
            assert result.get("result") == 5 or 5 in result.values()
        else:
            assert "5" in str(result)


@pytest.mark.asyncio
async def test_failing_tool_raises_tool_error():
    server = _build_server()
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        fail_tool = next(t for t in listing.tools if t.name == "fail")
        agentic_tool = mcp_tool_to_agentic_tool(fail_tool, session)

        with pytest.raises(ToolError, match="fail"):
            await agentic_tool.handler()


@pytest.mark.asyncio
async def test_echo_tool_returns_text_content():
    server = _build_server()
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        echo_tool = next(t for t in listing.tools if t.name == "echo")
        agentic_tool = mcp_tool_to_agentic_tool(echo_tool, session)
        result = await agentic_tool.handler(text="hello world")
        assert "hello world" in str(result)


def test_flatten_content_concatenates_text_blocks():
    from mcp.types import TextContent

    blocks = [
        TextContent(type="text", text="line1"),
        TextContent(type="text", text="line2"),
    ]
    assert _flatten_content(blocks) == "line1\nline2"


def test_flatten_content_handles_empty():
    assert _flatten_content([]) == ""


def test_call_result_to_tool_result_error():
    from mcp.types import TextContent

    result = call_result_to_tool_result(
        [TextContent(type="text", text="oops")],
        is_error=True,
    )
    assert result.ok is False
    assert "oops" in (result.error or "")


def test_call_result_to_tool_result_success():
    from mcp.types import TextContent

    result = call_result_to_tool_result(
        [TextContent(type="text", text="42")],
        is_error=False,
    )
    assert result.ok is True
    assert result.value == "42"
