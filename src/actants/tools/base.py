from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from actants.errors import ActantsError


class ToolError(ActantsError):
    """A tool could not be registered, or refused to run."""


class ToolResult(BaseModel):
    ok: bool = True
    value: Any = None
    error: str | None = None


def serialize_tool_result(result: ToolResult) -> str:
    """Render a ToolResult as the JSON string fed back to the model.

    Tool handlers are user code and may return anything — objects whose ``__str__``
    raises, or structures containing reference cycles. Serialization must never be able
    to abort the agent loop, so an unrenderable value degrades to a JSON error payload
    the model can actually read and react to.
    """
    if not result.ok:
        return json.dumps({"error": result.error})
    try:
        return json.dumps(result.value, default=str)
    except (TypeError, ValueError, RecursionError) as exc:
        return json.dumps(
            {
                "error": (
                    f"Tool returned a value that could not be serialized to JSON: "
                    f"{type(exc).__name__}: {exc}"
                )
            }
        )


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
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
