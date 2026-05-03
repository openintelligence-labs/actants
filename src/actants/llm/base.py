from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResult(BaseModel):
    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """Provider-agnostic tool description passed to an LLM for function calling."""

    name: str
    description: str
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})


class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallDelta(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class UsageDelta(BaseModel):
    type: Literal["usage"] = "usage"
    usage: TokenUsage
    cost_usd: float = 0.0


class FinishDelta(BaseModel):
    type: Literal["finish"] = "finish"
    reason: str | None = None


StreamEvent = TextDelta | ToolCallDelta | UsageDelta | FinishDelta


class BaseLLMProvider(ABC):
    name: str

    supports_tool_calls: bool = False
    supports_streaming_tools: bool = False

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        """Run a chat completion and return the result."""

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Stream text content chunks as they arrive. Does not emit tool-call deltas."""

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed events (text deltas, tool calls, usage, finish).

        Default implementation wraps ``stream`` for providers that only stream text.
        Providers with native streaming tool calls should override this.
        """
        async for chunk in self.stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            if chunk:
                yield TextDelta(text=chunk)
        yield FinishDelta(reason=None)

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is reachable."""
