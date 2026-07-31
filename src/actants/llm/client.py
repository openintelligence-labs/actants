from __future__ import annotations

import json
from collections.abc import AsyncIterator
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from actants.cache.protocol import CacheBackend
from actants.cost.tracker import CostTracker
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
)
from actants.llm.ollama import OllamaProvider
from actants.llm.partial_json import parse_partial_json
from actants.policies.retry import RetryPolicy, retry_async
from actants.tools.base import serialize_tool_result
from actants.tracing.otel import llm_span

if TYPE_CHECKING:
    from actants.tools.registry import ToolRegistry

T = TypeVar("T", bound=BaseModel)


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACTANTS_", extra="ignore")

    provider: str = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.7


def _make_provider(settings: LLMSettings) -> BaseLLMProvider:
    provider = settings.provider.lower()
    if provider == "ollama":
        return OllamaProvider(base_url=settings.base_url)
    if provider == "openai":
        from actants.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=settings.api_key)
    if provider == "anthropic":
        from actants.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=settings.api_key)
    if provider == "gemini":
        from actants.llm.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=settings.api_key)
    if provider == "groq":
        from actants.llm.groq_provider import GroqProvider

        return GroqProvider(api_key=settings.api_key)
    if provider == "mistral":
        from actants.llm.mistral_provider import MistralProvider

        return MistralProvider(api_key=settings.api_key)
    raise ValueError(f"Unknown provider: {settings.provider}")


class LLM:
    """High-level LLM client.

    Defaults to Ollama at localhost:11434. Accepts optional ``cache``, ``cost_tracker``,
    and ``retry_policy`` to layer observability and reliability on top of any provider.

    Example:
        >>> llm = LLM()  # ollama
        >>> r = await llm.complete("hello")
        >>> print(r.content, r.cost_usd)
    """

    def __init__(
        self,
        provider: BaseLLMProvider | str | None = None,
        *,
        model: str | None = None,
        settings: LLMSettings | None = None,
        cache: CacheBackend | None = None,
        cost_tracker: CostTracker | None = None,
        retry_policy: RetryPolicy | None = None,
        tracing: bool = True,
    ) -> None:
        self.settings = settings or LLMSettings()
        if model is not None:
            self.settings.model = model
        if isinstance(provider, str):
            self.settings.provider = provider
            provider = _make_provider(self.settings)
        elif provider is not None and not isinstance(provider, BaseLLMProvider):
            raise TypeError(
                "provider must be a BaseLLMProvider instance or a provider name string "
                f"(e.g. 'ollama', 'openai'), got {type(provider).__name__!r}"
            )
        self.provider = provider or _make_provider(self.settings)
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.retry_policy = retry_policy
        self.tracing = tracing

    async def complete(
        self,
        prompt: str | list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tag: str | None = None,
        use_cache: bool = True,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        """Run a chat completion and return the result.

        Passes through ``cache``, ``cost_tracker``, retry, and tracing layers set on the client.
        ``tag`` is recorded on the CostTracker for grouped reporting.
        """
        messages = self._normalize(prompt, system=system)
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        cache_key: str | None = None
        semantic_cache = (
            getattr(self.cache, "get_by_messages", None) if self.cache is not None else None
        )
        if semantic_cache is not None and use_cache and not tools:
            cached = await self.cache.get_by_messages(  # type: ignore[union-attr]
                messages, effective_model, effective_temp
            )
            if cached is not None:
                return cached
        elif self.cache is not None and use_cache and not tools:
            from actants.cache.memory import make_key

            cache_key = make_key(
                messages,
                effective_model,
                effective_temp,
                provider=self.provider.name,
                max_tokens=max_tokens,
                tools=tools,
            )
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        async def _call() -> CompletionResult:
            return await self.provider.complete(
                messages=messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=max_tokens,
                tools=tools,
            )

        result = await self._run(_call, op="complete", model=effective_model)

        if self.cost_tracker is not None:
            self.cost_tracker.record(result, tag=tag)
        if self.cache is not None and use_cache and not tools:
            if semantic_cache is not None:
                await self.cache.set_by_messages(  # type: ignore[union-attr]
                    messages, effective_model, effective_temp, result
                )
            elif cache_key is not None:
                await self.cache.set(cache_key, result)
        return result

    async def run_agent(
        self,
        prompt: str | list[ChatMessage],
        tools: ToolRegistry,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_steps: int = 6,
        system: str | None = None,
        tag: str | None = None,
    ) -> CompletionResult:
        """Run a tool-calling loop until the model returns a final text answer.

        Iterates up to ``max_steps`` times: completion -> dispatch tool calls -> feed
        results back as tool messages. Returns the final CompletionResult (content field
        holds the final answer; earlier tool calls are flushed into the message history).
        Raises RuntimeError if the loop exceeds ``max_steps`` without a final answer.
        """
        messages = self._normalize(prompt, system=system)
        specs = tools.as_specs()

        last: CompletionResult | None = None
        for _ in range(max_steps):
            last = await self.complete(
                messages,
                model=model,
                temperature=temperature,
                tag=tag,
                use_cache=False,
                tools=specs,
            )
            if not last.tool_calls:
                return last
            messages.append(
                ChatMessage(role="assistant", content=last.content, tool_calls=last.tool_calls)
            )
            for call in last.tool_calls:
                result = await tools.call(call.name, **call.arguments)
                payload = serialize_tool_result(result)
                messages.append(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
        raise RuntimeError(f"Agent loop exceeded max_steps={max_steps} without a final answer")

    async def extract(
        self,
        prompt: str | list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        system: str | None = None,
        max_repairs: int = 1,
    ) -> T:
        """Prompt the model and parse the response into the given pydantic model.

        If the first response doesn't parse, retries once with the parser error appended
        to the conversation so the model can self-correct. Works across all providers
        since it asks for JSON via prompt rather than provider-specific JSON modes.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        guide = (
            "Respond with ONLY valid JSON matching this JSON Schema. No prose, no code fences.\n"
            f"Schema:\n{schema_json}"
        )
        effective_system = f"{system}\n\n{guide}" if system else guide
        messages = self._normalize(prompt, system=effective_system)

        last_err: Exception | None = None
        for attempt in range(max_repairs + 1):
            result = await self.complete(
                messages,
                model=model,
                temperature=temperature,
                use_cache=False,
            )
            try:
                return schema.model_validate_json(_extract_json(result.content))
            except Exception as exc:
                last_err = exc
                if attempt >= max_repairs:
                    break
                messages.append(ChatMessage(role="assistant", content=result.content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"That did not parse as JSON matching the schema: {exc}. "
                            "Return ONLY corrected JSON."
                        ),
                    )
                )
        raise ValueError(f"Failed to extract {schema.__name__} from model output: {last_err}")

    async def extract_stream(
        self,
        prompt: str | list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> AsyncIterator[T]:
        """Yield progressively-complete pydantic objects as the model streams JSON.

        Each yielded instance represents the best parse of the bytes seen so far.
        Emits a new instance only when the parsed output has changed. The final
        yield is the fully-parsed and validated ``schema`` instance.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        guide = (
            "Respond with ONLY valid JSON matching this JSON Schema. No prose, no code fences.\n"
            f"Schema:\n{schema_json}"
        )
        effective_system = f"{system}\n\n{guide}" if system else guide
        messages = self._normalize(prompt, system=effective_system)

        buf = ""
        last_serialized: str | None = None
        async for event in self.provider.stream_events(
            messages=messages,
            model=model or self.settings.model,
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_tokens=None,
            tools=None,
        ):
            if isinstance(event, TextDelta):
                buf += event.text
                parsed = parse_partial_json(buf)
                if parsed is None:
                    continue
                try:
                    candidate = schema.model_validate(parsed)
                except Exception:
                    continue
                serialized = candidate.model_dump_json()
                if serialized != last_serialized:
                    last_serialized = serialized
                    yield candidate
        final = parse_partial_json(buf)
        if final is None:
            raise ValueError(f"Stream ended with no parseable JSON for {schema.__name__}: {buf!r}")
        try:
            candidate = schema.model_validate(final)
        except Exception as exc:
            raise ValueError(f"Stream ended with invalid {schema.__name__}: {exc}") from exc
        serialized = candidate.model_dump_json()
        if serialized != last_serialized:
            yield candidate

    async def stream(
        self,
        prompt: str | list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        messages = self._normalize(prompt, system=system)
        async for chunk in self.provider.stream(
            messages=messages,
            model=model or self.settings.model,
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    async def stream_events(
        self,
        prompt: str | list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield typed StreamEvents (text, tool_call, usage, finish)."""
        messages = self._normalize(prompt, system=system)
        async for event in self.provider.stream_events(
            messages=messages,
            model=model or self.settings.model,
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_tokens=max_tokens,
            tools=tools,
        ):
            yield event

    async def run_agent_stream(
        self,
        prompt: str | list[ChatMessage],
        tools: ToolRegistry,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_steps: int = 6,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming agent loop. Yields text deltas while the model is thinking,
        dispatches tool calls as they complete, and loops until a final text answer."""
        messages = self._normalize(prompt, system=system)
        specs = tools.as_specs()
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        for _ in range(max_steps):
            step_tool_calls = []
            step_text_parts: list[str] = []
            async for event in self.provider.stream_events(
                messages=messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=None,
                tools=specs,
            ):
                if isinstance(event, TextDelta):
                    step_text_parts.append(event.text)
                    yield event
                elif isinstance(event, ToolCallDelta):
                    step_tool_calls.append(event.tool_call)
                    yield event
                elif isinstance(event, FinishDelta):
                    pass
                else:
                    yield event
            if not step_tool_calls:
                yield FinishDelta(reason="stop")
                return
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="".join(step_text_parts),
                    tool_calls=step_tool_calls,
                )
            )
            for call in step_tool_calls:
                result = await tools.call(call.name, **call.arguments)
                payload = serialize_tool_result(result)
                messages.append(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
        raise RuntimeError(f"Agent stream exceeded max_steps={max_steps}")

    async def health(self) -> bool:
        return await self.provider.health()

    async def _run(self, coro_fn, *, op: str, model: str) -> CompletionResult:
        call = coro_fn
        if self.retry_policy is not None:
            call = partial(retry_async, coro_fn, self.retry_policy)

        if not self.tracing:
            return await call()

        async with llm_span(f"llm.{op}", provider=self.provider.name, model=model) as span:
            result = await call()
            span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
            span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
            span.set_attribute("llm.cost_usd", result.cost_usd)
            span.set_attribute("llm.latency_ms", result.latency_ms)
            return result

    @staticmethod
    def _normalize(prompt: str | list[ChatMessage], *, system: str | None) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        if isinstance(prompt, str):
            messages.append(ChatMessage(role="user", content=prompt))
        else:
            messages.extend(prompt)
        return messages


def _extract_json(text: str) -> str:
    """Pull JSON out of a model response that may include ```json fences or leading prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    start = s.find("{")
    start_arr = s.find("[")
    if start_arr != -1 and (start == -1 or start_arr < start):
        start = start_arr
    if start == -1:
        return s
    return s[start:]
