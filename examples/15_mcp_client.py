"""Consume MCP servers as agent tools. Two-line integration.

Requires:
    pip install -e ".[mcp]"
    # Plus a real MCP server you want to use, e.g.:
    uv tool install mcp-server-git

Run:
    python examples/15_mcp_client.py
"""

from __future__ import annotations

import asyncio

from agentic_kit import LLM, Agent
from agentic_kit.mcp import MCPClient
from agentic_kit.tools.registry import ToolRegistry


async def main() -> None:
    # Configure MCP servers using the same shape as Claude Desktop's mcpServers.
    # Stdio: spawn a subprocess. HTTP: hit a remote URL.
    servers = {
        "git": {"command": "uvx", "args": ["mcp-server-git", "--repository", "."]},
    }

    async with MCPClient(servers) as mcp:
        registry = ToolRegistry()
        for tool in mcp.tools():
            registry.register(tool)

        print(f"Loaded {len(registry.list())} tools from {mcp.server_names}:")
        for tool in registry.list()[:5]:
            print(f"  - {tool.name}: {tool.description[:60]}")

        agent = Agent(llm=LLM(model="llama3.2"), tools=registry)
        result = await agent.run("What is the git status of this repo? Use the git__status tool.")
        print("\nFinal:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
