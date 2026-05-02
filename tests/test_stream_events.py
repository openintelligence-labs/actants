from __future__ import annotations

import httpx
import pytest

from actants.llm.base import (
    ChatMessage,
    FinishDelta,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_stream_events_emits_text_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            '{"message":{"content":"hel"}}\n'
            '{"message":{"content":"lo"}}\n'
            '{"message":{"content":""},"done":true,"prompt_eval_count":3,'
            '"eval_count":2,"done_reason":"stop"}\n'
        )
        return httpx.Response(200, content=body.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = OllamaProvider(client=client)
        events = [
            e
            async for e in p.stream_events(
                messages=[ChatMessage(role="user", content="hi")], model="llama3.2"
            )
        ]

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "hello"
    usage = next(e for e in events if isinstance(e, UsageDelta))
    assert usage.usage.prompt_tokens == 3
    assert usage.usage.completion_tokens == 2
    finish = next(e for e in events if isinstance(e, FinishDelta))
    assert finish.reason == "stop"


@pytest.mark.asyncio
async def test_ollama_stream_events_emits_tool_calls():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            '{"message":{"content":"","tool_calls":[{"id":"c1","function":'
            '{"name":"get_time","arguments":{}}}]}}\n'
            '{"message":{"content":""},"done":true,"prompt_eval_count":5,'
            '"eval_count":1,"done_reason":"tool_calls"}\n'
        )
        return httpx.Response(200, content=body.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = OllamaProvider(client=client)
        events = [
            e
            async for e in p.stream_events(
                messages=[ChatMessage(role="user", content="time?")],
                model="llama3.2",
                tools=[
                    ToolSpec(
                        name="get_time",
                        description="get current time",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
            )
        ]

    tool_events = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.name == "get_time"
    assert tool_events[0].tool_call.id == "c1"
