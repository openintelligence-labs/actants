from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypedDict

from actants.mcp.adapters import mcp_tool_to_agentic_tool
from actants.mcp.transports import open_session
from actants.tools.base import Tool

if TYPE_CHECKING:
    from mcp import ClientSession


class MCPServerConfig(TypedDict, total=False):
    """Shape of one entry in an MCP servers config.

    Matches Claude Desktop's ``mcpServers`` shape so users can paste their existing config.
    """

    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str
    url: str
    headers: dict[str, str]


class MCPToolset:
    """Live tools loaded from one MCP server."""

    def __init__(self, name: str, session: ClientSession, tools: list[Tool]) -> None:
        self.name = name
        self.session = session
        self.tools = tools


class MCPClient:
    """Connect to N MCP servers, expose their tools to an Agent.

    Use as an async context manager — sessions stay open for the duration of the block.
    Each tool is name-prefixed with its server (``git__status``) to disambiguate.

    Example::

        async with MCPClient({
            "git": {"command": "uvx", "args": ["mcp-server-git"]},
            "fs":  {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
        }) as mcp:
            agent = Agent(llm=LLM(), tools=mcp.tools())
            await agent.run("show git status of the current repo")
    """

    def __init__(self, servers: dict[str, MCPServerConfig | dict[str, Any]]) -> None:
        if not isinstance(servers, dict):
            raise TypeError(
                "MCPClient expects a dict mapping server name -> config, got "
                f"{type(servers).__name__!r}. Example:\n"
                '    MCPClient({"git": {"command": "uvx", "args": ["mcp-server-git"]}})'
            )
        self._configs = dict(servers)
        self._stack: AsyncExitStack | None = None
        self._toolsets: dict[str, MCPToolset] = {}

    async def __aenter__(self) -> MCPClient:
        self._stack = AsyncExitStack()
        for name, config in self._configs.items():
            session = await self._stack.enter_async_context(open_session(config))
            listing = await session.list_tools()
            tools = [mcp_tool_to_agentic_tool(t, session, server_name=name) for t in listing.tools]
            self._toolsets[name] = MCPToolset(name=name, session=session, tools=tools)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._toolsets.clear()

    def tools(self) -> list[Tool]:
        """Return all tools from all connected servers as one flat list."""
        result: list[Tool] = []
        for ts in self._toolsets.values():
            result.extend(ts.tools)
        return result

    def toolset(self, name: str) -> MCPToolset:
        """Get tools from one named server only."""
        if name not in self._toolsets:
            raise KeyError(f"No MCP server named {name!r}. Known: {sorted(self._toolsets)}")
        return self._toolsets[name]

    @property
    def server_names(self) -> list[str]:
        return list(self._toolsets)
