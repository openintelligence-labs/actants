"""Roundtrip tests for actants.mcp.serve / build_server."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from actants.agents import Agent  # noqa: E402
from actants.llm.client import LLM  # noqa: E402
from actants.mcp import build_server  # noqa: E402
from actants.testing import FakeLLMProvider  # noqa: E402
from actants.tools.registry import ToolRegistry  # noqa: E402


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    async def echo(text: str) -> str:
        return text

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
    registry.register_function(
        "echo",
        "Echo back the input text",
        echo,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    return registry


@pytest.mark.asyncio
async def test_build_server_exposes_registry_tools():
    registry = _make_registry()
    server = build_server(registry, name="test-srv")

    async with create_connected_server_and_client_session(
        server._mcp_server,
    ) as session:
        listing = await session.list_tools()
        names = sorted(t.name for t in listing.tools)
        assert names == ["add", "echo"]


@pytest.mark.asyncio
async def test_server_dispatches_call_to_registry():
    registry = _make_registry()
    server = build_server(registry, name="test-srv")

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool("add", arguments={"a": 7, "b": 8})
        assert result.isError is False
        # Result is in content[0].text — FastMCP serializes our JSON return as text.
        text = result.content[0].text
        assert "15" in text


@pytest.mark.asyncio
async def test_server_propagates_tool_failure():
    registry = ToolRegistry()

    async def boom() -> str:
        raise RuntimeError("kaboom")

    registry.register_function("boom", "Always fails", boom)
    server = build_server(registry, name="test-srv")

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool("boom", arguments={})
        # Either MCP marks it as error, or the handler returns {"error": ...} JSON.
        text = result.content[0].text if result.content else ""
        assert "kaboom" in text or result.isError


@pytest.mark.asyncio
async def test_build_server_from_agent():
    """An Agent with tools should be servable directly."""
    registry = _make_registry()
    agent = Agent(
        llm=LLM(provider=FakeLLMProvider(), model="fake"),
        tools=registry,
    )
    server = build_server(agent, name="agent-srv")

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        assert {t.name for t in listing.tools} == {"add", "echo"}


def test_serve_rejects_agent_without_tools():
    agent = Agent(llm=LLM(provider=FakeLLMProvider(), model="fake"))
    with pytest.raises(ValueError, match="no tools"):
        build_server(agent)


def test_serve_rejects_wrong_type():
    with pytest.raises(TypeError, match="Agent or ToolRegistry"):
        build_server("not an agent")  # type: ignore[arg-type]
