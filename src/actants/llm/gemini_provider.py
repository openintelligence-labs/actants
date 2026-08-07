from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from actants.cost.pricing import estimate_cost
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

log = structlog.get_logger(__name__)

#: Passthrough keys Gemini takes at the top level of the request body. Everything else
#: a caller passes is a sampling knob and belongs inside ``generationConfig``.
_TOP_LEVEL_FIELDS = frozenset({"safetySettings", "cachedContent", "toolConfig"})


class GeminiProvider(BaseLLMProvider):
    """Google Gemini via its native REST API. No SDK dependency — just httpx.

    Pass ``api_key`` directly or set ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` in env.
    Default model should be passed via LLM(model=...).
    """

    name = "gemini"
    supports_tool_calls = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        import os

        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GeminiProvider requires an API key "
                "(api_key=..., GEMINI_API_KEY, or GOOGLE_API_KEY)"
            )
        self.base_url = base_url.rstrip("/")
        self._external = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(
                f"{self.base_url}/models", params={"key": self.api_key}, timeout=5.0
            )
            return r.status_code == 200
        except Exception as exc:
            log.debug("gemini_health_failed", error=str(exc))
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
        payload = self._build_payload(messages, temperature, max_tokens, tools=tools, **kwargs)
        url = f"{self.base_url}/models/{model}:generateContent"
        start = time.perf_counter()
        r = await self._client.post(url, params={"key": self.api_key}, json=payload)
        r.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        data = r.json()

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for cand in data.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []) or []:
                if "text" in part:
                    content_parts.append(part["text"])
                fn = part.get("functionCall")
                if fn:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fn.get("name", ""),
                            arguments=fn.get("args", {}) or {},
                        )
                    )
            break  # first candidate only

        usage_meta = data.get("usageMetadata", {}) or {}
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        finish_reason = None
        if data.get("candidates"):
            finish_reason = data["candidates"][0].get("finishReason")

        return CompletionResult(
            content="".join(content_parts),
            model=model,
            provider=self.name,
            usage=usage,
            cost_usd=estimate_cost(self.name, model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
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
        payload = self._build_payload(messages, temperature, max_tokens, tools=tools, **kwargs)
        url = f"{self.base_url}/models/{model}:streamGenerateContent"
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str | None = None
        async with self._client.stream(
            "POST",
            url,
            params={"key": self.api_key, "alt": "sse"},
            json=payload,
        ) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if not body:
                    continue
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for cand in data.get("candidates", []) or []:
                    for part in cand.get("content", {}).get("parts", []) or []:
                        if "text" in part:
                            yield TextDelta(text=part["text"])
                        fn = part.get("functionCall")
                        if fn:
                            yield ToolCallDelta(
                                tool_call=ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    name=fn.get("name", ""),
                                    arguments=fn.get("args", {}) or {},
                                )
                            )
                    if cand.get("finishReason"):
                        finish_reason = cand["finishReason"]
                usage_meta = data.get("usageMetadata", {}) or {}
                if usage_meta:
                    prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
                    completion_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        yield UsageDelta(
            usage=usage,
            cost_usd=estimate_cost(self.name, model, prompt_tokens, completion_tokens),
        )
        yield FinishDelta(reason=finish_reason)

    def _build_payload(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            if m.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": m.name or "tool",
                                    "response": {"result": m.content},
                                }
                            }
                        ],
                    }
                )
                continue
            role = "model" if m.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for tc in m.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": role, "parts": parts})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        # Provider-specific passthrough from LLM.complete(**extra). Gemini keeps its
        # sampling knobs (topP, topK, seed, stopSequences, ...) inside
        # ``generationConfig``; a caller naming a real top-level field gets it there.
        for key, value in kwargs.items():
            if key in _TOP_LEVEL_FIELDS:
                payload[key] = value
            else:
                payload["generationConfig"][key] = value
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]
        return payload
