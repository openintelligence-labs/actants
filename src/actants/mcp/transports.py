from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def open_session(config: dict[str, Any]):
    """Open an MCP ClientSession for one server config dict.

    Config shape mirrors Claude Desktop's ``mcpServers`` entries:
      - stdio: ``{"command": "uvx", "args": [...], "env": {...}}``
      - HTTP:  ``{"url": "https://...", "headers": {...}}``
    """
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise ImportError("MCP support requires `pip install actants[mcp]`") from exc

    if "command" in config:
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
            cwd=config.get("cwd"),
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session
    elif "url" in config:
        async with (
            streamablehttp_client(
                url=config["url"],
                headers=config.get("headers"),
            ) as (read, write, _get_session_id),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session
    else:
        raise ValueError(
            "MCP server config must contain either 'command' (stdio) or 'url' (HTTP). "
            f"Got: {sorted(config)}"
        )
