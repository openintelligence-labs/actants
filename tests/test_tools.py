from __future__ import annotations

import pytest

from actants.tools.base import Tool, ToolError
from actants.tools.registry import ToolRegistry


async def add(a: int, b: int) -> int:
    return a + b


async def broken() -> None:
    raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_register_and_call():
    registry = ToolRegistry()
    registry.register_function("add", "adds numbers", add)
    result = await registry.call("add", a=2, b=3)
    assert result.ok
    assert result.value == 5


@pytest.mark.asyncio
async def test_duplicate_registration_raises():
    registry = ToolRegistry()
    registry.register_function("add", "d", add)
    with pytest.raises(ValueError):
        registry.register_function("add", "d", add)


def test_unknown_tool_raises_from_get():
    """get() is the programmatic lookup, so an unknown name is a real error there."""
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.get("ghost")


@pytest.mark.asyncio
async def test_unknown_tool_from_call_is_reported_not_raised():
    """call() takes model-supplied names, so a hallucinated tool is a ToolResult."""
    registry = ToolRegistry()
    result = await registry.call("ghost")
    assert not result.ok
    assert "ghost" in (result.error or "")


@pytest.mark.asyncio
async def test_handler_error_captured():
    registry = ToolRegistry()
    registry.register_function("broken", "d", broken)
    result = await registry.call("broken")
    assert not result.ok
    assert "kaboom" in (result.error or "")


@pytest.mark.asyncio
async def test_permission_check_denies():
    async def deny(name: str, kwargs: dict) -> bool:
        return False

    registry = ToolRegistry(permission_check=deny)
    registry.register_function("add", "d", add, requires_permission=True)
    result = await registry.call("add", a=1, b=1)
    assert not result.ok
    assert "Permission denied" in (result.error or "")


@pytest.mark.asyncio
async def test_permission_check_allows():
    seen: dict = {}

    async def allow(name: str, kwargs: dict) -> bool:
        seen["name"] = name
        seen["kwargs"] = kwargs
        return True

    registry = ToolRegistry(permission_check=allow)
    registry.register_function("add", "d", add, requires_permission=True)
    result = await registry.call("add", a=4, b=5)
    assert result.ok
    assert result.value == 9
    assert seen["name"] == "add"
    assert seen["kwargs"] == {"a": 4, "b": 5}


@pytest.mark.asyncio
async def test_tool_without_handler_raises():
    t = Tool(name="noop", description="")
    with pytest.raises(ToolError):
        await t.call()
