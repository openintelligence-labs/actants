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
    #: Whether re-running this tool with the same arguments is harmless.
    #:
    #: Only consulted on `resume`, and only for the
    #: single call that was in flight when a run died — every call with a recorded
    #: result is replayed from the checkpoint and never re-dispatched. For that one
    #: ambiguous call, ``True`` means actants may re-dispatch it; ``False`` means it
    #: raises `UnresolvedToolCallError` and lets the caller
    #: decide.
    #:
    #: **The default is wrong for anything that writes.** It is ``True`` because most
    #: tools are reads, not because re-running is generally safe: mark ``send_email``,
    #: ``charge_card``, and every other externally-visible write ``idempotent=False``.
    idempotent: bool = True
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
