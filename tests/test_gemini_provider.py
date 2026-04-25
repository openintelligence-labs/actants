from __future__ import annotations

import httpx
import pytest

from agentic_kit.llm.base import ChatMessage, ToolSpec
from agentic_kit.llm.gemini_provider import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_complete_parses_response():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello back"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 5,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = GeminiProvider(api_key="fake-key", client=client)
        r = await p.complete(
            messages=[ChatMessage(role="user", content="hi")],
            model="gemini-2.5-flash",
        )
        assert r.content == "hello back"
        assert r.usage.prompt_tokens == 12
        assert r.usage.completion_tokens == 5
        assert r.provider == "gemini"
        assert r.cost_usd > 0  # gemini-2.5-flash has pricing
        assert "generateContent" in captured["url"]
        assert "fake-key" in captured["url"]


@pytest.mark.asyncio
async def test_gemini_tool_calls_parsed():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"city": "Tokyo"},
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 10,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = GeminiProvider(api_key="fake-key", client=client)
        r = await p.complete(
            messages=[ChatMessage(role="user", content="Weather in Tokyo?")],
            model="gemini-2.5-flash",
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            ],
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].arguments == {"city": "Tokyo"}


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GeminiProvider()


def test_groq_and_mistral_subclass_openai():
    from agentic_kit.llm.groq_provider import GroqProvider
    from agentic_kit.llm.mistral_provider import MistralProvider
    from agentic_kit.llm.openai_provider import OpenAIProvider

    g = GroqProvider(api_key="fake")
    m = MistralProvider(api_key="fake")
    assert isinstance(g, OpenAIProvider)
    assert isinstance(m, OpenAIProvider)
    assert g.name == "groq"
    assert m.name == "mistral"
