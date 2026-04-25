from __future__ import annotations

import httpx
import pytest

from agentic_kit.llm.base import ChatMessage, ToolSpec
from agentic_kit.llm.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_passes_tools_in_payload_and_parses_tool_calls():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 4,
                "done_reason": "tool_calls",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = OllamaProvider(client=client)
        result = await p.complete(
            messages=[ChatMessage(role="user", content="Weather in Paris?")],
            model="llama3.2",
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            ],
        )

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}
    assert tc.id == "call_123"
    assert "tools" in captured["body"]
    assert "get_weather" in captured["body"]


@pytest.mark.asyncio
async def test_ollama_serializes_assistant_tool_calls_in_followup():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "done"},
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    from agentic_kit.llm.base import ToolCall

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = OllamaProvider(client=client)
        await p.complete(
            messages=[
                ChatMessage(role="user", content="Weather in Paris?"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            name="get_weather",
                            arguments={"city": "Paris"},
                        )
                    ],
                ),
                ChatMessage(
                    role="tool",
                    content='{"temp_c": 18}',
                    tool_call_id="call_123",
                ),
            ],
            model="llama3.2",
        )
    assert "tool_calls" in captured["body"]
    assert "get_weather" in captured["body"]
