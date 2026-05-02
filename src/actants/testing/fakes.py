from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from actants.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


def fake_completion(
    content: str,
    *,
    model: str = "fake",
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 1,
    completion_tokens: int = 1,
) -> CompletionResult:
    """Build a CompletionResult for tests without remembering all the fields."""
    return CompletionResult(
        content=content,
        model=model,
        provider="fake",
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        tool_calls=tool_calls or [],
    )


def fake_tool_call_completion(
    tool_name: str, arguments: dict[str, Any], *, call_id: str = "tc1", model: str = "fake"
) -> CompletionResult:
    """Shortcut for a completion that requests a single tool call."""
    return fake_completion(
        content="",
        model=model,
        tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments)],
    )


class FakeLLMProvider(BaseLLMProvider):
    """Returns scripted responses; records every call for assertions.

    Pass a list of CompletionResult objects; each call to ``complete`` pops the next.
    Inspect ``calls`` (messages each call saw) and ``tools_seen`` (tool specs each call
    saw) to assert prompt construction in tests.
    """

    name = "fake"
    supports_tool_calls = True

    def __init__(self, responses: list[CompletionResult] | None = None) -> None:
        self._responses: list[CompletionResult] = list(responses or [])
        self.calls: list[list[ChatMessage]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []

    def queue(self, *results: CompletionResult) -> None:
        self._responses.extend(results)

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if not self._responses:
            return fake_completion("(no scripted response)", model=model)
        return self._responses.pop(0)

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        result = await self.complete(messages, model, temperature, max_tokens)
        for char in result.content:
            yield char

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        from actants.llm.base import FinishDelta, ToolCallDelta, UsageDelta

        result = await self.complete(messages, model, temperature, max_tokens, tools=tools)
        for char in result.content:
            yield TextDelta(text=char)
        for tc in result.tool_calls:
            yield ToolCallDelta(tool_call=tc)
        yield UsageDelta(usage=result.usage, cost_usd=0.0)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic fake embeddings: each text → a fixed-length vector seeded by hash."""

    name = "fake"

    def __init__(self, *, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, model: str = "fake-embed") -> EmbeddingResult:
        self.calls.append(list(texts))
        vectors = [self._vector_for(t) for t in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=model,
            provider=self.name,
            dimensions=self.dimensions,
        )

    def _vector_for(self, text: str) -> list[float]:
        seed = abs(hash(text))
        return [((seed >> (i * 4)) & 0xFF) / 255.0 - 0.5 for i in range(self.dimensions)]

    async def health(self) -> bool:
        return True
