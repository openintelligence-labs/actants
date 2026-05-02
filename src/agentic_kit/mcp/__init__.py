"""Model Context Protocol (MCP) integration for agentic-kit.

Two-line client::

    async with MCPClient({"git": {"command": "uvx", "args": ["mcp-server-git"]}}) as mcp:
        agent = Agent(llm=LLM(), tools=mcp.tools())

Two-line server (added in Day 2)::

    from agentic_kit.mcp import serve
    serve(agent, transport="stdio")

Requires the official MCP SDK: ``pip install agentic-kit[mcp]``.
"""

from __future__ import annotations

from agentic_kit.mcp.client import MCPClient, MCPServerConfig, MCPToolset
from agentic_kit.mcp.server import build_server, serve

__all__ = ["MCPClient", "MCPServerConfig", "MCPToolset", "build_server", "serve"]
