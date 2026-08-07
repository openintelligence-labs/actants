from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.errors import raise_for_ollama_error

log = structlog.get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    name = "ollama"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception as exc:
            log.debug("ollama_health_failed", error=str(exc))
            return False

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
        payload = self._build_payload(
            messages, model, temperature, max_tokens, stream=False, tools=tools, **kwargs
        )
        start = time.perf_counter()
        try:
            r = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
        except Exception as exc:
            await raise_for_ollama_error(
                exc, client=self._client, base_url=self.base_url, model=model
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        data = r.json()

        usage = TokenUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            total_tokens=int(data.get("prompt_eval_count", 0) or 0)
            + int(data.get("eval_count", 0) or 0),
        )

        message = data.get("message", {}) or {}
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", {}) or {},
                )
            )

        return CompletionResult(
            content=message.get("content", "") or "",
            model=model,
            provider=self.name,
            usage=usage,
            cost_usd=0.0,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
            tool_calls=tool_calls,
        )

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
        import json

        payload = self._build_payload(
            messages, model, temperature, max_tokens, stream=True, tools=tools, **kwargs
        )
        total_prompt = 0
        total_completion = 0
        finish_reason: str | None = None
        try:
            async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    # Body is needed to tell "model not pulled" from other 404s, and
                    # the stream is being abandoned anyway.
                    await r.aread()
                    r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = chunk.get("message", {}) or {}
                    content = message.get("content")
                    if content:
                        yield TextDelta(text=content)
                    for tc in message.get("tool_calls", []) or []:
                        fn = tc.get("function", {}) or {}
                        yield ToolCallDelta(
                            tool_call=ToolCall(
                                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                name=fn.get("name", ""),
                                arguments=fn.get("arguments", {}) or {},
                            )
                        )
                    if chunk.get("done"):
                        total_prompt = int(chunk.get("prompt_eval_count", 0) or 0)
                        total_completion = int(chunk.get("eval_count", 0) or 0)
                        finish_reason = chunk.get("done_reason")
        except Exception as exc:
            await raise_for_ollama_error(
                exc, client=self._client, base_url=self.base_url, model=model
            )
            raise
        yield UsageDelta(
            usage=TokenUsage(
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_prompt + total_completion,
            ),
            cost_usd=0.0,
        )
        yield FinishDelta(reason=finish_reason)

    def _build_payload(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int | None,
        *,
        stream: bool,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> dict:
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": model,
            "messages": [_message_to_ollama(m) for m in messages],
            "stream": stream,
            "options": options,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        return payload


def _message_to_ollama(m: ChatMessage) -> dict:
    out: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in m.tool_calls
        ]
    return out
