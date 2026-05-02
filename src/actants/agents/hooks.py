from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from actants.llm.base import ChatMessage, CompletionResult, ToolCall


@dataclass
class AgentHooks:
    """Optional async callbacks invoked during agent execution.

    All hooks are optional. Exceptions in hooks propagate — if you want best-effort,
    catch inside the hook. Hooks are called in order: before_step -> on_tool_call*
    (one per dispatched tool) -> after_step. on_error fires for any unhandled
    exception inside a step or tool call.
    """

    before_step: Callable[[int, list[ChatMessage]], Awaitable[None]] | None = None
    after_step: Callable[[int, CompletionResult], Awaitable[None]] | None = None
    on_tool_call: Callable[[ToolCall, Any], Awaitable[None]] | None = None
    on_error: Callable[[Exception], Awaitable[None]] | None = None
