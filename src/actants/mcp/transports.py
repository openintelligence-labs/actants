from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
from typing import Any


class MCPConnectionError(RuntimeError):
    """An MCP server could not be started or reached."""


def _describe(exc: BaseException) -> str:
    """Flatten an ExceptionGroup down to its most informative leaf."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


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
        raise ImportError(
            "MCP support requires the optional `mcp` dependency. "
            "Install it with `pip install 'actants[mcp]'`."
        ) from exc

    if not isinstance(config, dict):
        raise TypeError(
            "Each MCP server config must be a dict, got "
            f"{type(config).__name__!r}. Example: "
            '{"git": {"command": "uvx", "args": ["mcp-server-git"]}}.'
        )

    if "command" in config:
        argv = " ".join([str(config["command"]), *(str(a) for a in config.get("args", []))])
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
            cwd=config.get("cwd"),
        )
        cm = stdio_client(params)
        try:
            streams = await cm.__aenter__()
        except BaseException as exc:  # noqa: BLE001 — re-raised as MCPConnectionError below
            raise MCPConnectionError(
                f"Cannot start MCP server: command {config['command']!r} could not be run. "
                f"The configured command line is `{argv}`. "
                "Install the server binary (many MCP servers run via `uvx`, which needs "
                "`pip install uv`), or correct the 'command' entry in your server config. "
                f"(underlying error: {_describe(exc)})"
            ) from exc
    elif "url" in config:
        cm = streamablehttp_client(url=config["url"], headers=config.get("headers"))
        try:
            streams = await cm.__aenter__()
        except BaseException as exc:  # noqa: BLE001 — re-raised as MCPConnectionError below
            raise MCPConnectionError(
                f"Cannot reach the MCP server at {config['url']!r}. "
                "Check that the server is running and the URL is correct (streamable-HTTP "
                "MCP endpoints usually end in `/mcp`). "
                f"(underlying error: {_describe(exc)})"
            ) from exc
    else:
        raise ValueError(
            "MCP server config must contain either 'command' (stdio) or 'url' (HTTP). "
            f"Got: {sorted(config)}"
        )

    read, write = streams[0], streams[1]
    try:
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
            except BaseException as exc:  # noqa: BLE001 — re-raised as MCPConnectionError below
                target = config.get("url") or config.get("command")
                raise MCPConnectionError(
                    f"Connected to the MCP server at {target!r} but the MCP handshake failed. "
                    "The process started but did not speak the Model Context Protocol — "
                    "check that the command is an MCP server and not an ordinary program. "
                    f"(underlying error: {_describe(exc)})"
                ) from exc
            yield session
    finally:
        # Teardown noise must not mask the real error the caller is handling.
        with contextlib.suppress(Exception):
            await cm.__aexit__(None, None, None)
