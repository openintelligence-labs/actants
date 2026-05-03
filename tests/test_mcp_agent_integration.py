"""End-to-end: MCPClient → Tool → Agent loop with FakeLLM.

These tests don't spawn subprocesses — they use the SDK's in-memory transport but
exercise the full MCPClient lifecycle and Agent integration.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from actants.agents import Agent  # noqa: E402
from actants.llm.client import LLM  # noqa: E402
from actants.mcp.adapters import mcp_tool_to_agentic_tool  # noqa: E402
from actants.testing import (  # noqa: E402
    FakeLLMProvider,
    fake_completion,
    fake_tool_call_completion,
)
from actants.tools.registry import ToolRegistry  # noqa: E402


def _server_with_calculator() -> FastMCP:
    server = FastMCP("calc")

    @server.tool()
    async def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    return server


@pytest.mark.asyncio
async def test_agent_dispatches_mcp_tool_via_fake_llm():
    """Full loop: agent sees MCP tool spec, FakeLLM scripts a tool call, MCP runs it."""
    server = _server_with_calculator()

    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(
            create_connected_server_and_client_session(server._mcp_server)
        )
        listing = await session.list_tools()
        registry = ToolRegistry()
        for mcp_tool in listing.tools:
            registry.register(mcp_tool_to_agentic_tool(mcp_tool, session))

        provider = FakeLLMProvider(
            [
                fake_tool_call_completion("multiply", {"a": 6, "b": 7}, call_id="t1"),
                fake_completion("The answer is 42."),
            ]
        )
        agent = Agent(
            llm=LLM(provider=provider, model="fake"),
            tools=registry,
        )
        result = await agent.run("What is 6 times 7?")

    assert result.content == "The answer is 42."
    assert len(result.steps) == 2
    assert result.steps[0].tool_calls[0].name == "multiply"
    # MCP returns structured output as JSON; payload contains 42 in some form
    payload = result.steps[0].tool_results[0]
    parsed: Any = json.loads(payload) if payload.startswith(("{", "[")) else payload
    assert "42" in str(parsed)


@pytest.mark.asyncio
async def test_mcp_tools_carry_input_schema_to_agent():
    """The agent's tool spec list must include the MCP tool's JSON Schema."""
    server = _server_with_calculator()
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listing = await session.list_tools()
        registry = ToolRegistry()
        for mcp_tool in listing.tools:
            registry.register(mcp_tool_to_agentic_tool(mcp_tool, session))

        specs = registry.as_specs()
        multiply_spec = next(s for s in specs if s.name == "multiply")
        assert multiply_spec.parameters["type"] == "object"
        assert "a" in multiply_spec.parameters["properties"]
        assert "b" in multiply_spec.parameters["properties"]
