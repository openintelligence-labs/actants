from __future__ import annotations

import pytest

from agentic_kit.testing import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    fake_completion,
    fake_tool_call_completion,
)


@pytest.mark.asyncio
async def test_fake_llm_pops_responses_in_order():
    fake = FakeLLMProvider([fake_completion("first"), fake_completion("second")])
    a = await fake.complete([], model="x")
    b = await fake.complete([], model="x")
    assert a.content == "first"
    assert b.content == "second"


@pytest.mark.asyncio
async def test_fake_llm_records_calls():
    from agentic_kit.llm.base import ChatMessage

    fake = FakeLLMProvider([fake_completion("ok")])
    msgs = [ChatMessage(role="user", content="hi")]
    await fake.complete(msgs, model="x", tools=None)
    assert fake.calls == [msgs]
    assert fake.tools_seen == [None]


@pytest.mark.asyncio
async def test_fake_llm_returns_default_when_empty():
    fake = FakeLLMProvider()
    result = await fake.complete([], model="x")
    assert "no scripted response" in result.content


@pytest.mark.asyncio
async def test_fake_llm_streams_chars():
    fake = FakeLLMProvider([fake_completion("hi")])
    chars = [c async for c in fake.stream([], model="x")]
    assert chars == ["h", "i"]


def test_fake_tool_call_completion_helper():
    result = fake_tool_call_completion("add", {"a": 1, "b": 2})
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_fake_embedding_provider_dimensions_match():
    fake = FakeEmbeddingProvider(dimensions=16)
    result = await fake.embed(["hello", "world"])
    assert result.dimensions == 16
    assert all(len(v) == 16 for v in result.vectors)
