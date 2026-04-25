from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentic_kit.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
)
from agentic_kit.policies.fallback import AllProvidersFailedError, FallbackProvider


class FailingProvider(BaseLLMProvider):
    name = "failing"

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        raise RuntimeError("boom")

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream boom")
        yield  # unreachable but keeps this an async generator

    async def health(self) -> bool:
        return False


class OkProvider(BaseLLMProvider):
    name = "ok"

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        return CompletionResult(
            content="ok",
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        for c in ("a", "b"):
            yield c

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_fallback_uses_second_when_first_fails():
    provider = FallbackProvider([(FailingProvider(), None), (OkProvider(), None)])
    result = await provider.complete([ChatMessage(role="user", content="hi")], "m")
    assert result.content == "ok"
    assert result.provider == "ok"


@pytest.mark.asyncio
async def test_fallback_all_fail_raises():
    provider = FallbackProvider([(FailingProvider(), None), (FailingProvider(), None)])
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await provider.complete([ChatMessage(role="user", content="hi")], "m")
    assert len(exc_info.value.errors) == 2


@pytest.mark.asyncio
async def test_fallback_stream_uses_second_when_first_fails():
    provider = FallbackProvider([(FailingProvider(), None), (OkProvider(), None)])
    chunks = [c async for c in provider.stream([ChatMessage(role="user", content="hi")], "m")]
    assert chunks == ["a", "b"]


@pytest.mark.asyncio
async def test_fallback_respects_per_provider_model():
    seen: dict = {}

    class RecordingProvider(OkProvider):
        name = "recording"

        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
            seen["model"] = model
            return await super().complete(messages, model, temperature, max_tokens, **kwargs)

    provider = FallbackProvider([(RecordingProvider(), "override-model")])
    await provider.complete([ChatMessage(role="user", content="hi")], "caller-requested")
    assert seen["model"] == "override-model"


def test_fallback_requires_nonempty_chain():
    with pytest.raises(ValueError):
        FallbackProvider([])
