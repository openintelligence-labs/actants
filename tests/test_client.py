from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
)
from actants.llm.client import LLM, LLMSettings


class FakeProvider(BaseLLMProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], str, float]] = []

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> CompletionResult:
        self.calls.append((list(messages), model, temperature))
        return CompletionResult(
            content="echo:" + messages[-1].content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        for part in ("a", "b", "c"):
            yield part

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_complete_with_string_prompt():
    fake = FakeProvider()
    llm = LLM(provider=fake, model="fake-1")
    result = await llm.complete("hello", system="you are a test")
    assert result.content == "echo:hello"
    assert result.model == "fake-1"
    assert len(fake.calls) == 1
    messages, model, _ = fake.calls[0]
    assert messages[0].role == "system"
    assert messages[0].content == "you are a test"
    assert messages[1].role == "user"
    assert messages[1].content == "hello"
    assert model == "fake-1"


@pytest.mark.asyncio
async def test_complete_with_message_list():
    fake = FakeProvider()
    llm = LLM(provider=fake, model="fake-1")
    await llm.complete(
        [
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="q2"),
        ]
    )
    messages, _, _ = fake.calls[0]
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert [m.content for m in messages] == ["q1", "a1", "q2"]


@pytest.mark.asyncio
async def test_stream_passthrough():
    fake = FakeProvider()
    llm = LLM(provider=fake)
    chunks = [c async for c in llm.stream("hi")]
    assert chunks == ["a", "b", "c"]


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("ACTANTS_MODEL", "custom-model")
    monkeypatch.setenv("ACTANTS_TEMPERATURE", "0.2")
    s = LLMSettings()
    assert s.model == "custom-model"
    assert s.temperature == 0.2


def test_provider_accepts_name_string():
    from actants.llm.ollama import OllamaProvider

    llm = LLM(provider="ollama", model="gemma4:latest")
    assert isinstance(llm.provider, OllamaProvider)
    assert llm.settings.provider == "ollama"
    assert llm.settings.model == "gemma4:latest"


def test_provider_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown provider: not-a-provider"):
        LLM(provider="not-a-provider")


def test_provider_wrong_type_raises_type_error():
    with pytest.raises(TypeError, match="provider name string"):
        LLM(provider=42)  # type: ignore[arg-type]


def test_provider_instance_still_accepted():
    fake = FakeProvider()
    llm = LLM(provider=fake)
    assert llm.provider is fake
