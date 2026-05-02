from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any, Literal

from actants.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from actants.agents.agent import Agent


def build_server(
    source: Agent | ToolRegistry,
    *,
    name: str | None = None,
    instructions: str | None = None,
) -> FastMCP:
    """Build a FastMCP server that exposes the tools from an Agent or ToolRegistry.

    The server registers each tool with its existing name, description, and JSON Schema.
    Calling a tool delegates to the live ToolRegistry, so permission checks and other
    registry behavior continue to apply.

    This is the lower-level builder. For one-line setup use ``serve()``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("MCP support requires `pip install actants[mcp]`") from exc

    registry = _registry_from(source)
    server = FastMCP(name=name or "actants", instructions=instructions)

    for tool in registry.list():
        _register_tool(server, registry, tool.name, tool.description, tool.input_schema or {})

    return server


def serve(
    source: Agent | ToolRegistry,
    *,
    transport: Literal["stdio", "streamable-http", "sse"] = "stdio",
    name: str | None = None,
    instructions: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run an MCP server exposing the tools from an Agent or ToolRegistry.

    One-line server::

        from actants.mcp import serve
        serve(agent)                                  # stdio
        serve(agent, transport="streamable-http")     # HTTP on 127.0.0.1:8000

    Blocks until the transport is closed. For stdio, that means until the parent
    process disconnects; for HTTP, until the process is killed. Use ``build_server``
    if you want to embed the FastMCP instance in a larger ASGI app.
    """
    server = build_server(source, name=name, instructions=instructions)
    if transport in ("streamable-http", "sse"):
        server.settings.host = host
        server.settings.port = port
    server.run(transport=transport)


def _registry_from(source: Agent | ToolRegistry) -> ToolRegistry:
    if isinstance(source, ToolRegistry):
        return source
    # Lazy-import Agent to avoid circular import.
    from actants.agents.agent import Agent as AgentClass

    if isinstance(source, AgentClass):
        if source.tools is None:
            raise ValueError(
                "Cannot serve an Agent with no tools — pass tools=ToolRegistry(...) "  # noqa: E501
                "when building it."
            )
        return source.tools
    raise TypeError(f"serve() expects an Agent or ToolRegistry; got {type(source).__name__}")


_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _register_tool(
    server: FastMCP,
    registry: ToolRegistry,
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> None:
    """Register one tool on a FastMCP server, dispatching to the live registry.

    FastMCP introspects the handler's signature to build its tool schema. We
    synthesize a real signature from ``input_schema`` so each property becomes a
    named parameter — otherwise FastMCP would expose a single ``kwargs: dict`` arg
    and the LLM would have to wrap calls.
    """
    properties: dict[str, dict[str, Any]] = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])

    params: list[inspect.Parameter] = []
    for prop_name, prop_schema in properties.items():
        py_type: Any = _JSON_TO_PY.get(prop_schema.get("type", "string"), Any)
        default: Any = inspect.Parameter.empty if prop_name in required else None
        params.append(
            inspect.Parameter(
                prop_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=py_type,
            )
        )

    async def handler(**kwargs: Any) -> str:
        # Drop unset optional params so registry handlers see only what was supplied.
        kwargs = {k: v for k, v in kwargs.items() if k in properties}
        result = await registry.call(name, **kwargs)
        if not result.ok:
            return json.dumps({"error": result.error})
        if isinstance(result.value, str):
            return result.value
        return json.dumps(result.value, default=str)

    handler.__name__ = name
    handler.__doc__ = description
    handler.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=params, return_annotation=str
    )
    handler.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}

    if not asyncio.iscoroutinefunction(handler):  # pragma: no cover - defensive
        raise RuntimeError("MCP tool handler must be async")

    server.add_tool(handler, name=name, description=description)
