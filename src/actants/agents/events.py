"""Streaming event taxonomy for ``Agent.stream()``.

Mirrors and extends ``actants.llm.base`` stream events with agent-level
lifecycle markers. All events are dataclass-shaped for ``match`` ergonomics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from actants.llm.base import CompletionResult, ToolCall


@dataclass
class AgentTextDelta:
    """Token-level text from the model. Append to a buffer to display live."""

    text: str
    step: int


@dataclass
class AgentToolCallStarted:
    """Model has emitted a complete tool call; dispatch is about to begin."""

    call: ToolCall
    step: int


@dataclass
class AgentToolCallCompleted:
    """Tool finished executing. ``ok`` reflects whether it raised."""

    call: ToolCall
    value: Any
    ok: bool
    step: int


@dataclass
class AgentStepCompleted:
    """One LLM call + (optionally) one round of tool dispatch finished."""

    step: int
    completion: CompletionResult


@dataclass
class AgentRunCompleted:
    """The agent reached a final answer. ``content`` holds it."""

    content: str
    final: CompletionResult
