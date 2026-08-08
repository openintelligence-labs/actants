from __future__ import annotations

import json
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
from actants.llm.finish_reason import normalize_finish_reason
from actants.llm.structured import NativeSchemaMode

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIProvider(BaseLLMProvider):
    name = "openai"
    supports_tool_calls = True
    supports_streaming_tools = True
    #: Subclasses in `openai_compatible` override this per provider —
    #: speaking the OpenAI wire format does not imply implementing ``json_schema``.
    native_schema_mode: NativeSchemaMode = "openai_json_schema"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install with `pip install actants[openai]` to use OpenAIProvider"
                ) from exc
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            client = AsyncOpenAI(**client_kwargs)
        self._client = client

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
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
        start = time.perf_counter()
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Provider-specific passthrough from LLM.complete(**extra) — seed, top_p,
            # stop, presence_penalty. Forwarded verbatim; the SDK rejects names it does
            # not know, which beats dropping them silently as this did before.
            **kwargs,
        }
        if tools:
            request_kwargs["tools"] = [_tool_to_openai(t) for t in tools]
        r = await self._client.chat.completions.create(**request_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = TokenUsage(
            prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
            completion_tokens=r.usage.completion_tokens if r.usage else 0,
            total_tokens=r.usage.total_tokens if r.usage else 0,
        )
        choice = r.choices[0]
        tool_calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return CompletionResult(
            content=choice.message.content or "",
            model=model,
            provider=self.name,
            usage=usage,
            cost_usd=estimate_cost(self.name, model, usage.prompt_tokens, usage.completion_tokens),
            latency_ms=latency_ms,
            # Subclasses (Groq, Mistral, xAI, ...) pass their own ``name`` here; none of
            # them has a table, so all of them resolve to the OpenAI vocabulary — which
            # is correct, because they return OpenAI's response shape verbatim.
            finish_reason=normalize_finish_reason(self.name, choice.finish_reason),
            raw_finish_reason=choice.finish_reason,
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
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,  # provider-specific passthrough; see complete()
        }
        if tools:
            request_kwargs["tools"] = [_tool_to_openai(t) for t in tools]
        stream = await self._client.chat.completions.create(**request_kwargs)

        pending: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        async for chunk in stream:
            if chunk.usage is not None:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                yield UsageDelta(
                    usage=usage,
                    cost_usd=estimate_cost(
                        self.name, model, usage.prompt_tokens, usage.completion_tokens
                    ),
                )
                continue
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                yield TextDelta(text=delta.content)
            for tc in delta.tool_calls or []:
                slot = pending.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason
                for slot in pending.values():
                    try:
                        args = json.loads(slot["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield ToolCallDelta(
                        tool_call=ToolCall(id=slot["id"] or "", name=slot["name"], arguments=args)
                    )
                pending.clear()
        yield FinishDelta.from_provider(self.name, finish_reason)


def _message_to_openai(m: ChatMessage) -> dict[str, Any]:
    base: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.name:
        base["name"] = m.name
    if m.tool_call_id and m.role == "tool":
        base["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        base["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    return base


def _tool_to_openai(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }
