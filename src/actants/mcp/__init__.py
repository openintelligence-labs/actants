"""Model Context Protocol (MCP) integration for actants.

Two-line client::

    async with MCPClient({"git": {"command": "uvx", "args": ["mcp-server-git"]}}) as mcp:
        agent = Agent(llm=LLM(), tools=mcp.tools())

Two-line server (added in Day 2)::

    from actants.mcp import serve
    serve(agent, transport="stdio")

Requires the official MCP SDK: ``pip install actants[mcp]``.
"""

from __future__ import annotations

from actants.mcp.client import MCPClient, MCPServerConfig, MCPToolset
from actants.mcp.server import build_server, serve

__all__ = ["MCPClient", "MCPServerConfig", "MCPToolset", "build_server", "serve"]
