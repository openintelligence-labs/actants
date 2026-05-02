from __future__ import annotations

import httpx
import pytest

from actants.llm.base import ChatMessage
from actants.llm.ollama import OllamaProvider


def _fake_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_health_ok():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=_fake_transport(handler)) as client:
        p = OllamaProvider(client=client)
        assert await p.health() is True


@pytest.mark.asyncio
async def test_health_unreachable():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    async with httpx.AsyncClient(transport=_fake_transport(handler)) as client:
        p = OllamaProvider(client=client)
        assert await p.health() is False


@pytest.mark.asyncio
async def test_complete_parses_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = request.content.decode()
        assert "llama3.2" in body
        assert "hello" in body
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hi there"},
                "prompt_eval_count": 5,
                "eval_count": 7,
                "done_reason": "stop",
            },
        )

    async with httpx.AsyncClient(transport=_fake_transport(handler)) as client:
        p = OllamaProvider(client=client)
        result = await p.complete(
            messages=[ChatMessage(role="user", content="hello")],
            model="llama3.2",
        )
        assert result.content == "hi there"
        assert result.provider == "ollama"
        assert result.model == "llama3.2"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 7
        assert result.usage.total_tokens == 12
        assert result.cost_usd == 0.0
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_stream_yields_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            '{"message":{"content":"hel"}}\n'
            '{"message":{"content":"lo"}}\n'
            '{"message":{"content":" world"}}\n'
            '{"done":true}\n'
        )
        return httpx.Response(200, content=body.encode())

    async with httpx.AsyncClient(transport=_fake_transport(handler)) as client:
        p = OllamaProvider(client=client)
        chunks = [
            c
            async for c in p.stream(
                messages=[ChatMessage(role="user", content="hi")], model="llama3.2"
            )
        ]
        assert "".join(chunks) == "hello world"
