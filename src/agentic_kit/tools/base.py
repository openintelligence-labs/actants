from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolError(Exception):
    pass


class ToolResult(BaseModel):
    ok: bool = True
    value: Any = None
    error: str | None = None


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)
    requires_permission: bool = False
    handler: Callable[..., Awaitable[Any]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, **kwargs: Any) -> ToolResult:
        if self.handler is None:
            raise ToolError(f"Tool {self.name} has no handler bound")
        try:
            value = await self.handler(**kwargs)
            return ToolResult(ok=True, value=value)
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
