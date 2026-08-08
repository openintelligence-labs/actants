from __future__ import annotations

import builtins
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from actants.llm.base import ToolSpec
from actants.tools.base import Tool, ToolError, ToolResult

_JSON_SCHEMA_TYPES = {
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
}

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _validate_input_schema(name: str, schema: dict[str, Any]) -> None:
    """Reject schemas the LLM providers will silently choke on."""
    if "type" not in schema:
        raise ValueError(
            f"input_schema for tool {name!r} is missing the required 'type' key. "
            'A tool schema must be a JSON Schema object, e.g. {"type": "object", '
            '"properties": {"a": {"type": "integer"}}, "required": ["a"]}.'
        )
    declared = schema["type"]
    if declared not in _JSON_SCHEMA_TYPES:
        raise ValueError(
            f"input_schema for tool {name!r} declares type {declared!r}, which is not a "
            f"valid JSON Schema type. Valid types: {', '.join(sorted(_JSON_SCHEMA_TYPES))}. "
            'Tool schemas are almost always {"type": "object", ...}.'
        )
    if declared == "object" and "properties" not in schema:
        raise ValueError(
            f"input_schema for tool {name!r} is an object but has no 'properties' key. "
            'Use {"type": "object", "properties": {}} for a tool that takes no arguments.'
        )


def _schema_from_signature(name: str, handler: Callable[..., Awaitable[Any]]) -> dict[str, Any]:
    """Derive a JSON Schema from an async handler's type annotations.

    Raises if a parameter is un-annotated — an un-annotated tool is invisible to the
    model, which is a silent failure worth stopping at registration time.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exotic callables
        raise ValueError(
            f"Cannot inspect the signature of the handler for tool {name!r}; "
            "pass input_schema=... explicitly."
        ) from exc

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    missing: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            missing.append(param_name)
            continue
        json_type = _PY_TO_JSON_TYPE.get(annotation)
        if json_type is None and isinstance(annotation, str):
            json_type = _PY_TO_JSON_TYPE.get(
                {"str": str, "int": int, "float": float, "bool": bool}.get(annotation, object)
            )
        if json_type is None:
            missing.append(param_name)
            continue
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    if missing:
        raise ValueError(
            f"Cannot build an input schema for tool {name!r}: parameter(s) "
            f"{', '.join(repr(m) for m in missing)} have no supported type annotation. "
            "Annotate them with str/int/float/bool/list/dict, or pass input_schema=... "
            "explicitly. Without a schema the model cannot see the tool's arguments."
        )

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry with optional permission callback invoked before each call."""

    def __init__(
        self,
        permission_check: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._permission_check = permission_check

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError(
                f"register() expects a Tool instance, got {type(tool).__name__!r}. "
                "To register a plain async function use register_function(name, "
                "description, handler)."
            )
        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}. "
                "Tool names must be unique within a registry; pick a different name "
                "or build a second ToolRegistry."
            )
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[Any]],
        *,
        input_schema: dict[str, Any] | None = None,
        requires_permission: bool = False,
        idempotent: bool = True,
    ) -> Tool:
        """Register an async function as a tool.

        When ``input_schema`` is omitted it is derived from the handler's type
        annotations. Every parameter must be annotated, otherwise the model would
        receive a tool with no visible arguments.

        Pass ``idempotent=False`` for anything with an externally-visible side effect —
        see `idempotent>`, which the
        default of ``True`` is wrong for.
        """
        if not callable(handler):
            raise TypeError(
                f"handler for tool {name!r} must be an async function, got "
                f"{type(handler).__name__!r}."
            )
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(
                f"handler for tool {name!r} must be an async function (`async def`), got a "
                "regular function. actants dispatches tools with `await`. "
                f"Change it to `async def {getattr(handler, '__name__', name)}(...)`, or wrap "
                "a blocking call with `asyncio.to_thread`."
            )

        if input_schema is None:
            input_schema = _schema_from_signature(name, handler)
        else:
            if not isinstance(input_schema, dict):
                raise TypeError(
                    f"input_schema for tool {name!r} must be a dict describing a JSON "
                    f"Schema, got {type(input_schema).__name__!r}."
                )
            _validate_input_schema(name, input_schema)

        tool = Tool(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema,
            requires_permission=requires_permission,
            idempotent=idempotent,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "<none registered>"
            raise ToolError(f"Unknown tool: {name}. Registered tools: {known}.")
        return self._tools[name]

    def list(self) -> builtins.list[Tool]:
        return list(self._tools.values())

    # `builtins.list` throughout: the `list` method above shadows the builtin inside the
    # class body, so a bare `list[...]` annotation on any method defined after it
    # resolves to the method object and is silently not a type.
    def as_specs(self) -> builtins.list[ToolSpec]:
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
        """Dispatch a tool call, returning failures as a ToolResult rather than raising.

        The tool name and arguments come from the model, so a hallucinated name or a
        bogus argument list is normal model behaviour — not a programming error. Both
        are reported back as ``ok=False`` so the agent loop can feed the error to the
        model and let it correct itself. Use `get` when you want an unknown tool
        to raise.
        """
        try:
            tool = self.get(name)
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        if tool.requires_permission and self._permission_check is not None:
            allowed = await self._permission_check(name, kwargs)
            if not allowed:
                return ToolResult(ok=False, error=f"Permission denied for tool {name}")
        return await tool.call(**kwargs)
