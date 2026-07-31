from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from actants.llm.base import (
    BaseLLMProvider,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    TokenUsage,
)
from actants.llm.client import LLM
from actants.llm.partial_json import parse_partial_json


class Report(BaseModel):
    title: str
    severity: str
    tags: list[str] = []


def test_parse_complete_json():
    assert parse_partial_json('{"a": 1}') == {"a": 1}


def test_parse_strips_code_fences():
    assert parse_partial_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_closes_open_object():
    result = parse_partial_json('{"title": "x", "severity": "low"')
    assert result == {"title": "x", "severity": "low"}


def test_parse_closes_open_array():
    result = parse_partial_json('{"tags": ["a", "b"')
    assert result == {"tags": ["a", "b"]}


def test_parse_handles_incomplete_string():
    result = parse_partial_json('{"title": "inc')
    # Trimming back to the prior boundary may leave an empty object or nothing;
    # both are acceptable, the contract is only that it must not raise.
    assert result is None or isinstance(result, dict)


def test_parse_returns_none_for_garbage():
    assert parse_partial_json("nope nothing here") is None


class ScriptedStreamProvider(BaseLLMProvider):
    """Provider that streams a pre-split sequence of text chunks."""

    name = "scripted_stream"

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
        return CompletionResult(
            content="".join(self._chunks),
            model=model,
            provider=self.name,
            usage=TokenUsage(),
        )

    async def stream(
        self, messages, model, temperature=0.7, max_tokens=None, **kw
    ) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kw
    ) -> AsyncIterator[StreamEvent]:
        for c in self._chunks:
            yield TextDelta(text=c)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_extract_stream_yields_progressive_objects():
    chunks = [
        '{"title": "Crash',
        ' bug", "severity":',
        ' "high", "tags": ["ui"',
        "]}",
    ]
    llm = LLM(provider=ScriptedStreamProvider(chunks), model="m", tracing=False)
    seen = [r async for r in llm.extract_stream("analyze this", Report)]
    assert seen  # at least one emission
    final = seen[-1]
    assert final.title == "Crash bug"
    assert final.severity == "high"
    assert final.tags == ["ui"]
