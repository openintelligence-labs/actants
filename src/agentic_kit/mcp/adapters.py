from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentic_kit.tools.base import Tool, ToolError, ToolResult

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as MCPTool


def mcp_tool_to_agentic_tool(
    mcp_tool: MCPTool,
    session: ClientSession,
    *,
    server_name: str | None = None,
) -> Tool:
    """Convert an MCP Tool description into an agentic-kit Tool.

    The returned Tool's handler dispatches over the live ClientSession.
    Tool name is prefixed with the server name when given (``git__status``)
    to disambiguate between MCP servers exposing same-named tools.
    """
    qualified_name = f"{server_name}__{mcp_tool.name}" if server_name else mcp_tool.name

    async def handler(**kwargs: Any) -> Any:
        result = await session.call_tool(mcp_tool.name, arguments=kwargs)
        if result.isError:
            text = _flatten_content(result.content)
            raise ToolError(f"MCP tool {mcp_tool.name!r} failed: {text}")
        if result.structuredContent is not None:
            return result.structuredContent
        return _flatten_content(result.content)

    return Tool(
        name=qualified_name,
        description=mcp_tool.description or f"MCP tool {mcp_tool.name}",
        input_schema=mcp_tool.inputSchema or {"type": "object", "properties": {}},
        handler=handler,
    )


def _flatten_content(content: list[Any]) -> str:
    """Flatten MCP content blocks into a single string for the LLM.

    Text blocks concatenate; non-text blocks are JSON-serialized as a fallback so
    the model sees structured information rather than a placeholder.
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        elif block_type == "resource":
            resource = getattr(block, "resource", None)
            uri = getattr(resource, "uri", "?") if resource else "?"
            parts.append(f"[resource: {uri}]")
        elif block_type == "image":
            mime = getattr(block, "mimeType", "image/?")
            parts.append(f"[image: {mime}]")
        else:
            try:
                parts.append(json.dumps(block.model_dump(), default=str))
            except Exception:
                parts.append(str(block))
    return "\n".join(parts)


def call_result_to_tool_result(content: list[Any], *, is_error: bool) -> ToolResult:
    """Convert an MCP CallToolResult into an agentic-kit ToolResult.

    Used by tests; the live adapter raises ToolError on failure rather than
    returning a ToolResult so it integrates with ToolRegistry's error path.
    """
    text = _flatten_content(content)
    if is_error:
        return ToolResult(ok=False, error=text)
    return ToolResult(ok=True, value=text)
