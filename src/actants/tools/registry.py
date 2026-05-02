from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from actants.llm.base import ToolSpec
from actants.tools.base import Tool, ToolError, ToolResult


class ToolRegistry:
    """Registry with optional permission callback invoked before each call."""

    def __init__(
        self,
        permission_check: Callable[[str, dict], Awaitable[bool]] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._permission_check = permission_check

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[Any]],
        *,
        input_schema: dict | None = None,
        requires_permission: bool = False,
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema or {"type": "object", "properties": {}},
            requires_permission=requires_permission,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def as_specs(self) -> list[ToolSpec]:
        """Return a provider-agnostic tool description list for LLM function calling."""
        return [
            ToolSpec(
                name=t.name,
                description=t.description,
                parameters=t.input_schema or {"type": "object", "properties": {}},
            )
            for t in self._tools.values()
        ]

    async def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if tool.requires_permission and self._permission_check is not None:
            allowed = await self._permission_check(name, kwargs)
            if not allowed:
                return ToolResult(ok=False, error=f"Permission denied for tool {name}")
        return await tool.call(**kwargs)
