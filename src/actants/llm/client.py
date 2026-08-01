from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from difflib import get_close_matches
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from actants.cache.protocol import CacheBackend
from actants.cache.request import CacheRequest
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
from actants.llm.errors import (
    MissingAPIKeyError,
    ProviderNotInstalledError,
    UnknownProviderError,
    tool_calls_not_supported,
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


#: provider name -> (env var holding its API key, extra that installs it)
_PROVIDER_REQUIREMENTS: dict[str, tuple[str | None, str]] = {
    "ollama": (None, "ollama"),
    "openai": ("OPENAI_API_KEY", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
    "gemini": ("GEMINI_API_KEY", "gemini"),
    "groq": ("GROQ_API_KEY", "groq"),
    "mistral": ("MISTRAL_API_KEY", "mistral"),
}

KNOWN_PROVIDERS: tuple[str, ...] = tuple(_PROVIDER_REQUIREMENTS)


def _resolve_api_key(provider: str, settings: LLMSettings) -> str | None:
    """Return the API key for ``provider``, raising an actionable error if absent."""
    env_var, extra = _PROVIDER_REQUIREMENTS[provider]
    if env_var is None:
        return None
    key = settings.api_key or os.environ.get(env_var)
    if not key:
        raise MissingAPIKeyError(
            f"Provider {provider!r} requires an API key and none was found. "
            f"Set the {env_var} environment variable, or pass it explicitly: "
            f"LLM(settings=LLMSettings(provider={provider!r}, api_key='...')). "
            "To run locally with no API key, use the default Ollama provider: LLM()."
        )
    return key


def _make_provider(settings: LLMSettings) -> BaseLLMProvider:
    provider = settings.provider.lower()
    if provider not in _PROVIDER_REQUIREMENTS:
        suggestion = get_close_matches(provider, KNOWN_PROVIDERS, n=1, cutoff=0.6)
        did_you_mean = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise UnknownProviderError(
            f"Unknown provider {settings.provider!r}.{did_you_mean} "
            f"Known providers: {', '.join(KNOWN_PROVIDERS)}."
        )

    api_key = _resolve_api_key(provider, settings)
    _, extra = _PROVIDER_REQUIREMENTS[provider]

    try:
        if provider == "ollama":
            return OllamaProvider(base_url=settings.base_url)
        if provider == "openai":
            from actants.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(api_key=api_key)
        if provider == "anthropic":
            from actants.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(api_key=api_key)
        if provider == "gemini":
            from actants.llm.gemini_provider import GeminiProvider

            return GeminiProvider(api_key=api_key)
        if provider == "groq":
            from actants.llm.groq_provider import GroqProvider

            return GroqProvider(api_key=api_key)
        from actants.llm.mistral_provider import MistralProvider

        return MistralProvider(api_key=api_key)
    except ImportError as exc:
        raise ProviderNotInstalledError(
            f"Provider {provider!r} needs an optional dependency that is not installed. "
            f"Install it with `pip install 'actants[{extra}]'`. "
            f"(underlying error: {exc})"
        ) from exc


def _require_registry(tools: object) -> None:
    if tools is None:
        raise TypeError(
            "tools is required and must be a ToolRegistry. "
            "For a plain completion with no tools use llm.complete(prompt) instead."
        )
    if not hasattr(tools, "as_specs"):
        raise TypeError(
            f"tools must be a ToolRegistry, got {type(tools).__name__!r}. "
            "Build one with:\n"
            "    registry = ToolRegistry()\n"
            "    registry.register_function('add', 'Add two integers', add)"
        )


def _require_tool_specs(tools: object) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise TypeError(
            f"tools must be a list of ToolSpec, got {type(tools).__name__!r}. "
            "To use a ToolRegistry, pass registry.as_specs() — or call "
            "llm.run_agent(prompt, registry), which does it for you."
        )
    for i, spec in enumerate(tools):
        if not isinstance(spec, ToolSpec):
            raise TypeError(
                f"tools[{i}] must be a ToolSpec, got {type(spec).__name__!r}. "
                "Build them with ToolRegistry(...).as_specs(), or construct one directly: "
                "ToolSpec(name='add', description='Add two integers', parameters={...})."
            )


def _require_pydantic_model(schema: object) -> None:
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"schema must be a pydantic BaseModel subclass, got {schema!r}. "
            "Define one with:\n"
            "    class Person(BaseModel):\n"
            "        name: str\n"
            "        age: int"
        )


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
        if settings is not None and not isinstance(settings, LLMSettings):
            raise TypeError(
                "settings must be an LLMSettings instance, got "
                f"{type(settings).__name__!r}. Build one with "
                "LLMSettings(provider='ollama', model='llama3.2'), or pass the common "
                "options directly: LLM(provider=..., model=...)."
            )
        self.settings = settings or LLMSettings()
        if model is not None:
            if not isinstance(model, str):
                raise TypeError(
                    f"model must be a string, got {type(model).__name__!r}. "
                    "Example: LLM(model='llama3.2')."
                )
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
        _require_tool_specs(tools)
        self._require_tool_support(tools)
        messages = self._normalize(prompt, system=system)
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        # One description of the request, shared by every cache backend. Both the
        # exact-match key and the semantic scope hash derive from it, so the two kinds of
        # backend can never disagree about what makes a request unique.
        cache_request = CacheRequest(
            messages=messages,
            model=effective_model,
            temperature=effective_temp,
            provider=self.provider.name,
            max_tokens=max_tokens,
            tools=tools,
        )
        request_cache = getattr(self.cache, "get_request", None) if self.cache is not None else None
        caching = self.cache is not None and use_cache and not tools
        if caching:
            if request_cache is not None:
                cached = await self.cache.get_request(cache_request)  # type: ignore[union-attr]
            else:
                cached = await self.cache.get(cache_request.key())  # type: ignore[union-attr]
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
        if caching:
            if request_cache is not None:
                await self.cache.set_request(cache_request, result)  # type: ignore[union-attr]
            else:
                await self.cache.set(cache_request.key(), result)  # type: ignore[union-attr]
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
        _require_registry(tools)
        messages = self._normalize(prompt, system=system)
        specs = tools.as_specs()
        self._require_tool_support(specs)

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
        _require_pydantic_model(schema)
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
        _require_pydantic_model(schema)
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
        _require_tool_specs(tools)
        self._require_tool_support(tools, streaming=True)
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
        _require_registry(tools)
        messages = self._normalize(prompt, system=system)
        specs = tools.as_specs()
        self._require_tool_support(specs, streaming=True)
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

    def _require_tool_support(
        self, tools: list[ToolSpec] | None, *, streaming: bool = False
    ) -> None:
        """Fail fast when tools are passed to a provider that cannot call them.

        Providers declare this with ``supports_tool_calls`` /
        ``supports_streaming_tools``. Without the check the specs are dropped somewhere
        in the provider's request builder and the model answers as though no tools were
        offered — a silent wrong answer rather than an error.
        """
        if not tools:
            return
        capable = (
            self.provider.supports_streaming_tools
            if streaming
            else self.provider.supports_tool_calls
        )
        if capable:
            return
        raise tool_calls_not_supported(
            self.provider.name,
            [t.name for t in tools],
            streaming=streaming,
        )

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
            return messages
        if not isinstance(prompt, list):
            raise TypeError(
                f"prompt must be a string or a list of ChatMessage, got {type(prompt).__name__!r}. "
                "Example: await llm.complete('hello') or "
                "await llm.complete([ChatMessage(role='user', content='hello')])."
            )
        for i, item in enumerate(prompt):
            if isinstance(item, ChatMessage):
                messages.append(item)
            elif isinstance(item, dict):
                # Accept the OpenAI-style dict shape, but validate it properly.
                try:
                    messages.append(ChatMessage.model_validate(item))
                except Exception as exc:
                    raise TypeError(
                        f"prompt[{i}] is a dict that is not a valid ChatMessage: {exc}"
                    ) from exc
            else:
                raise TypeError(
                    f"prompt[{i}] must be a ChatMessage (or a dict with 'role' and 'content'), "
                    f"got {type(item).__name__!r}. "
                    "Wrap plain strings: ChatMessage(role='user', content=...)."
                )
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
