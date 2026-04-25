from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from agentic_kit.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
)
from agentic_kit.llm.client import LLM


class Person(BaseModel):
    name: str
    age: int


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        self.calls.append(list(messages))
        content = self._responses.pop(0)
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_extract_parses_plain_json():
    provider = ScriptedProvider(['{"name": "Ada", "age": 36}'])
    llm = LLM(provider=provider, model="test")
    person = await llm.extract("Make up a person.", Person)
    assert person.name == "Ada"
    assert person.age == 36


@pytest.mark.asyncio
async def test_extract_handles_code_fences():
    provider = ScriptedProvider(['```json\n{"name": "Grace", "age": 85}\n```'])
    llm = LLM(provider=provider, model="test")
    person = await llm.extract("Make up a person.", Person)
    assert person.name == "Grace"


@pytest.mark.asyncio
async def test_extract_repairs_invalid_json():
    provider = ScriptedProvider(
        [
            "totally not json",
            '{"name": "Alan", "age": 41}',
        ]
    )
    llm = LLM(provider=provider, model="test")
    person = await llm.extract("Make up a person.", Person, max_repairs=1)
    assert person.name == "Alan"
    # Second call should include a repair request
    assert len(provider.calls) == 2
    assert any("did not parse" in m.content for m in provider.calls[1])


@pytest.mark.asyncio
async def test_extract_raises_after_all_repairs_fail():
    provider = ScriptedProvider(["nope", "still nope"])
    llm = LLM(provider=provider, model="test")
    with pytest.raises(ValueError, match="Failed to extract Person"):
        await llm.extract("Make up a person.", Person, max_repairs=1)
