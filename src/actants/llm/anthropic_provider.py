from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, api_key: str | None = None, client: AsyncAnthropic | None = None) -> None:
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Install with `pip install actants[anthropic]` to use AnthropicProvider"
                ) from exc
            client = AsyncAnthropic(api_key=api_key)
        self._client = client

    async def health(self) -> bool:
        try:
            await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    def _split_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        system: str | None = None
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "",
                                "content": m.content,
                            }
                        ],
                    }
                )
                continue
            if m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": m.role, "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

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
        system, msgs = self._split_messages(messages)
        start = time.perf_counter()
        request_kwargs: dict[str, Any] = {
            "model": model,
            "system": system or "",
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if tools:
            request_kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        r = await self._client.messages.create(**request_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in r.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                content_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        usage = TokenUsage(
            prompt_tokens=r.usage.input_tokens,
            completion_tokens=r.usage.output_tokens,
            total_tokens=r.usage.input_tokens + r.usage.output_tokens,
        )
        return CompletionResult(
            content="".join(content_parts),
            model=model,
            provider=self.name,
            usage=usage,
            cost_usd=estimate_cost(self.name, model, usage.prompt_tokens, usage.completion_tokens),
            latency_ms=latency_ms,
            finish_reason=r.stop_reason,
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
        import json as _json

        system, msgs = self._split_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "system": system or "",
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if tools:
            request_kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        tool_blocks: dict[int, dict[str, Any]] = {}
        final_usage: TokenUsage | None = None
        stop_reason: str | None = None
        async with self._client.messages.stream(**request_kwargs) as stream:
            async for raw in stream:
                etype = getattr(raw, "type", None)
                if etype == "content_block_start":
                    block = getattr(raw, "content_block", None)
                    if block is not None and getattr(block, "type", "") == "tool_use":
                        tool_blocks[raw.index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                        }
                elif etype == "content_block_delta":
                    delta = getattr(raw, "delta", None)
                    if delta is None:
                        continue
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dtype == "input_json_delta" and raw.index in tool_blocks:
                        tool_blocks[raw.index]["arguments"] += delta.partial_json
                elif etype == "content_block_stop":
                    slot = tool_blocks.pop(raw.index, None)
                    if slot is not None:
                        try:
                            args = _json.loads(slot["arguments"] or "{}")
                        except _json.JSONDecodeError:
                            args = {}
                        yield ToolCallDelta(
                            tool_call=ToolCall(id=slot["id"], name=slot["name"], arguments=args)
                        )
                elif etype == "message_delta":
                    delta = getattr(raw, "delta", None)
                    if delta is not None:
                        stop_reason = getattr(delta, "stop_reason", None) or stop_reason
                    usage = getattr(raw, "usage", None)
                    if usage is not None:
                        final_usage = TokenUsage(
                            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                            total_tokens=(getattr(usage, "input_tokens", 0) or 0)
                            + (getattr(usage, "output_tokens", 0) or 0),
                        )
        if final_usage is not None:
            yield UsageDelta(
                usage=final_usage,
                cost_usd=estimate_cost(
                    self.name,
                    model,
                    final_usage.prompt_tokens,
                    final_usage.completion_tokens,
                ),
            )
        yield FinishDelta(reason=stop_reason)
