from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from difflib import get_close_matches
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from actants.cache.protocol import CacheBackend, RequestCacheBackend
from actants.cache.request import CacheRequest
from actants.cost.tracker import CostTracker
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.errors import (
    MissingAPIKeyError,
    ProviderNotInstalledError,
    UnknownProviderError,
    tool_calls_not_supported,
)
from actants.llm.ollama import OllamaProvider
from actants.llm.partial_json import parse_partial_json
from actants.llm.structured import (
    ANTHROPIC_EXTRACT_TOOL,
    SchemaPlan,
    build_schema_plan,
    drop_defaulted_nulls,
    prompt_schema_guide,
)
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
#:
#: Everything from ``groq`` down speaks the OpenAI wire format and is served by
#: `actants.llm.openai_compatible`, so adding one is a row here plus a row there.
#: Each has its own extra in pyproject — all aliases for ``openai``, the SDK they
#: actually need — so the "install this extra" error can name what the user expects.
_PROVIDER_REQUIREMENTS: dict[str, tuple[str | None, str]] = {
    "ollama": (None, "ollama"),
    "openai": ("OPENAI_API_KEY", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
    "gemini": ("GEMINI_API_KEY", "gemini"),
    "groq": ("GROQ_API_KEY", "groq"),
    "mistral": ("MISTRAL_API_KEY", "mistral"),
    "xai": ("XAI_API_KEY", "xai"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek"),
    "together": ("TOGETHER_API_KEY", "together"),
    "fireworks": ("FIREWORKS_API_KEY", "fireworks"),
    "openrouter": ("OPENROUTER_API_KEY", "openrouter"),
    "cerebras": ("CEREBRAS_API_KEY", "cerebras"),
    "perplexity": ("PERPLEXITY_API_KEY", "perplexity"),
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
        # Everything else is OpenAI-compatible: same request path, different base URL.
        # Dispatching through the table rather than falling off the end of an if-chain
        # means a provider added to _PROVIDER_REQUIREMENTS but not to the compatible
        # table raises here, instead of being silently built as some other provider —
        # which is what the old trailing `return MistralProvider(...)` would have done.
        from actants.llm.openai_compatible import (
            OPENAI_COMPATIBLE_PROVIDERS,
            openai_compatible_provider,
        )

        entry = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if entry is None:  # pragma: no cover - a test asserts the two tables agree
            raise UnknownProviderError(
                f"Provider {provider!r} is registered but has no constructor. "
                "This is an actants bug: add it to OPENAI_COMPATIBLE_PROVIDERS, or "
                "give it its own branch in _make_provider."
            )
        return openai_compatible_provider(provider, *entry)(api_key=api_key)
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

    Settings lifetime
    -----------------
    ``settings`` is **copied** at construction, so mutating the object you passed in
    afterwards has no effect on this client — and a settings object shared between two
    clients cannot have one client's ``model=`` override leak into the other.

    Of the copy's own fields, ``model`` and ``temperature`` are re-read on every call, so
    assigning ``llm.settings.model`` does change subsequent requests. ``provider`` and
    ``base_url`` are read once, to build `provider`; assigning them later does
    nothing. Build a new ``LLM`` to change provider, or pass a provider instance
    directly.

    Counting conventions
    --------------------
    Three parameters bound how many times actants will go back to the model, and they do
    not all count the same thing. The rule is that a name saying *attempts* or *steps*
    bounds the **total**, and a name saying *repairs* bounds the **extras**:

    * ``RetryPolicy(max_attempts=N)`` — at most ``N`` requests in total. ``N=1`` means no
      retry.
    * ``max_steps=N`` (`run_agent`, `run_agent_stream`, ``Agent.run``) — at
      most ``N`` LLM round-trips in the loop, in total.
    * ``max_repairs=N`` (`extract`) — the initial completion, plus at most ``N``
      self-correction attempts after it: ``N + 1`` requests in total. ``N=0`` disables
      repair, and is the analogue of ``max_attempts=1``.

    ``max_repairs`` counts differently on purpose, because a repair is not a retry. A
    retry re-sends the same request after a transport failure; a repair sends a *new,
    longer* conversation containing the model's bad output and the parser error, so the
    model can correct itself. Naming it ``max_attempts`` would suggest ``1`` allows one
    self-correction, when it would in fact allow none.
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
        # Copied, not aliased: `model=` and `provider=` below write into this object, and
        # a caller who builds one LLMSettings and passes it to two clients must not have
        # the second client's overrides appear in the first's settings — or, worse, in
        # their own object after construction.
        self.settings = settings.model_copy(deep=True) if settings is not None else LLMSettings()
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
        self._last_schema_plan: SchemaPlan | None = None

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
        **extra: Any,
    ) -> CompletionResult:
        """Run a chat completion and return the result.

        Passes through ``cache``, ``cost_tracker``, retry, and tracing layers set on the client.
        ``tag`` is recorded on the CostTracker for grouped reporting.

        Any additional keyword arguments are provider-specific parameters — ``seed``,
        ``top_p``, ``stop``, ``presence_penalty`` — and are forwarded verbatim to the
        provider *and* folded into the cache key. Two requests differing only in
        ``seed=1`` versus ``seed=2`` are therefore different requests, as they must be::

            await llm.complete("hi", seed=42, top_p=0.9)

        Which names a provider accepts is the provider's business; actants does not
        validate them, so a name the provider does not recognise surfaces as that
        provider's own error.
        """
        _require_tool_specs(tools)
        self._require_tool_support(tools)
        messages = self._normalize(prompt, system=system)
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        # One description of the request, shared by every cache backend. Both the
        # exact-match key and the semantic scope hash derive from it, so the two kinds of
        # backend can never disagree about what makes a request unique.
        #
        # ``extra`` must be carried here as well as sent to the provider: it is exactly
        # the set of parameters that change the answer without appearing in any modelled
        # field, so omitting it would serve a seed=1 answer to a seed=2 request.
        cache_request = CacheRequest(
            messages=messages,
            model=effective_model,
            temperature=effective_temp,
            provider=self.provider.name,
            max_tokens=max_tokens,
            tools=tools,
            extra=extra,
        )
        # `self.cache` is narrowed directly rather than through a `caching` bool: mypy
        # cannot carry a None-check stored in a variable, and routing the two backend
        # protocols through `isinstance` (RequestCacheBackend is runtime_checkable)
        # replaces a getattr probe that had to be silenced with two type: ignores.
        caching = self.cache is not None and use_cache and not tools
        if self.cache is not None and caching:
            cached: CompletionResult | None
            if isinstance(self.cache, RequestCacheBackend):
                cached = await self.cache.get_request(cache_request)
            else:
                cached = await self.cache.get(cache_request.key())
            if cached is not None:
                return cached

        async def _call() -> CompletionResult:
            return await self.provider.complete(
                messages=messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=max_tokens,
                tools=tools,
                **extra,
            )

        result = await self._run(_call, op="complete", model=effective_model)

        if self.cost_tracker is not None:
            self.cost_tracker.record(result, tag=tag)
        if self.cache is not None and caching:
            if isinstance(self.cache, RequestCacheBackend):
                await self.cache.set_request(cache_request, result)
            else:
                await self.cache.set(cache_request.key(), result)
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
        tag: str | None = None,
        max_repairs: int = 1,
    ) -> T:
        """Prompt the model and parse the response into the given pydantic model.

        Where the provider supports provider-native constrained decoding — OpenAI's
        ``response_format``, a forced tool call on Anthropic, ``responseSchema`` on
        Gemini, ``format`` on Ollama — the schema is sent on the wire and invalid output
        is impossible rather than merely unlikely. Everywhere else, and for any schema
        the provider's dialect cannot express, the schema is described in a system
        prompt and a failed parse is repaired. Both paths return the same validated
        instance; `last_schema_plan` reports which one ran.

        ``tag`` is recorded on the CostTracker as in `complete`. Every attempt is
        recorded under the same tag, including repairs that failed to parse — a repair
        costs real tokens, so hiding it would understate what the extraction spent.

        ``max_repairs`` counts *repair* attempts, not total attempts: the initial
        completion always happens, and ``max_repairs=1`` (the default) allows one
        self-correction after it, for at most two requests. ``max_repairs=0`` disables
        repair entirely. This differs deliberately from ``RetryPolicy.max_attempts`` and
        ``max_steps``, which bound the total; see the note in the class docstring.
        Its meaning is unchanged on the native path — the repair loop is simply never
        entered, because a schema-valid response cannot fail to parse.

        Strict mode has no way to say "this field may be absent", only "it may be null",
        so a field with a non-null default is sent as nullable-and-required. A ``null``
        for one of those is read back as the absence it was standing in for, and the
        field's default applies — otherwise a provider obeying the schema exactly would
        produce a response pydantic rejects, and no amount of repair could fix it.
        """
        _require_pydantic_model(schema)
        plan = self._plan_schema(schema)
        messages = self._normalize(prompt, system=self._extract_system(schema, plan, system))

        last_err: Exception | None = None
        for attempt in range(max_repairs + 1):
            result = await self.complete(
                messages,
                model=model,
                temperature=temperature,
                tag=tag,
                use_cache=False,
                **plan.request_kwargs,
            )
            try:
                parsed = json.loads(_extract_payload(result, plan))
                return schema.model_validate(drop_defaulted_nulls(parsed, plan.nulls_mean_default))
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

    def _plan_schema(self, schema: type[BaseModel], *, streaming: bool = False) -> SchemaPlan:
        """Choose the transport for one extraction and record it for `last_schema_plan`."""
        plan = build_schema_plan(schema, self.provider.native_schema_mode, streaming=streaming)
        self._last_schema_plan = plan
        return plan

    def last_schema_plan(self) -> SchemaPlan | None:
        """How the most recent `extract` / `extract_stream` call was sent.

        ``None`` before the first call. This is the supported way to tell the native
        path from the prompt path — the alternative would be a log line on every
        extraction, which is noise in the overwhelming case where the answer never
        changes for a given provider and schema.

        Reflects one client's last call, so read it from the same task that made the
        call: two concurrent extractions on one ``LLM`` overwrite each other's plan.
        """
        return self._last_schema_plan

    def _extract_system(
        self, schema: type[BaseModel], plan: SchemaPlan, system: str | None
    ) -> str | None:
        """Append the JSON-Schema instruction to ``system``, unless the wire carries it.

        On the native path the schema is already a hard constraint, so repeating it in
        the prompt would spend tokens on every call to say something the decoder is
        enforcing anyway.
        """
        if plan.native:
            return system
        guide = prompt_schema_guide(schema)
        return f"{system}\n\n{guide}" if system else guide

    async def extract_stream(
        self,
        prompt: str | list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        system: str | None = None,
        tag: str | None = None,
    ) -> AsyncIterator[T]:
        """Yield progressively-complete pydantic objects as the model streams JSON.

        Each yielded instance represents the best parse of the bytes seen so far.
        Emits a new instance only when the parsed output has changed. The final
        yield is the fully-parsed and validated ``schema`` instance.

        Uses provider-native constrained decoding where the provider has a mode that
        streams as *text* — every partial parse then comes off a stream that cannot go
        schema-invalid. Anthropic's forced tool call is excluded: its JSON arrives as
        tool-call input rather than text deltas, so that provider streams via the prompt
        path. `last_schema_plan` reports which ran.

        ``tag`` is recorded on the CostTracker as in `stream_events`.
        """
        _require_pydantic_model(schema)
        plan = self._plan_schema(schema, streaming=True)
        messages = self._normalize(prompt, system=self._extract_system(schema, plan, system))

        buf = ""
        last_serialized: str | None = None
        async for event in self._stream_layered(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=None,
            tools=None,
            op="extract_stream",
            tag=tag,
            extra=plan.request_kwargs or None,
        ):
            if isinstance(event, TextDelta):
                buf += event.text
                parsed = parse_partial_json(buf)
                if parsed is None:
                    continue
                try:
                    candidate = schema.model_validate(
                        drop_defaulted_nulls(parsed, plan.nulls_mean_default)
                    )
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
            candidate = schema.model_validate(drop_defaulted_nulls(final, plan.nulls_mean_default))
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
        tag: str | None = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Stream plain text chunks, with the client's retry and tracing applied.

        ``tag`` is recorded on the CostTracker exactly as in `complete`, so cost
        attribution survives the switch from ``complete()`` to streaming. The spend is
        recorded once, when the provider reports usage — see `stream_events`.

        Extra keyword arguments are provider-specific parameters, forwarded verbatim —
        see `complete`.
        """
        messages = self._normalize(prompt, system=system)
        async for event in self._stream_layered(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
            op="stream",
            tag=tag,
            extra=extra,
        ):
            if isinstance(event, TextDelta):
                yield event.text

    async def stream_events(
        self,
        prompt: str | list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tag: str | None = None,
        tools: list[ToolSpec] | None = None,
        **extra: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Yield typed StreamEvents (text, tool_call, usage, finish).

        Applies the same retry and tracing layers as `complete`, and honours the
        same per-call ``model`` / ``temperature`` overrides. Extra keyword arguments are
        provider-specific parameters, forwarded verbatim — see `complete`.

        ``tag`` is recorded on the CostTracker just as in `complete`. A stream
        reports its spend in a single `UsageDelta` near the end,
        so the tracker is credited once per streamed request, at that point — a stream
        the consumer abandons before the usage event therefore records nothing, because
        actants never saw what it cost.
        """
        _require_tool_specs(tools)
        self._require_tool_support(tools, streaming=True)
        messages = self._normalize(prompt, system=system)
        async for event in self._stream_layered(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            op="stream_events",
            tag=tag,
            extra=extra,
        ):
            yield event

    async def _stream_layered(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[ToolSpec] | None,
        op: str,
        tag: str | None = None,
        record_cost: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """The single path every stream in actants goes through.

        Applies per-call model/temperature overrides, tracing, retry, and cost tracking,
        so a streamed run behaves like a non-streamed one. Retry is deliberately limited
        to failures that happen *before the first event reaches the consumer*: restarting
        after chunks have been yielded would splice two completions into one response,
        the same defect that was fixed in ``FallbackProvider.stream``.

        Cost is recorded here rather than in each caller so that every streaming entry
        point — `stream`, `stream_events`, `extract_stream`,
        `run_agent_stream` — attributes spend to ``tag`` through the same code
        ``complete`` uses. ``record_cost=False`` is for the one caller that records its
        own `CompletionResult` (`Agent.stream`), which
        would otherwise double-count.
        """
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        policy = self.retry_policy
        attempts = policy.max_attempts if policy is not None else 1

        async def _open() -> AsyncIterator[StreamEvent]:
            return self.provider.stream_events(
                messages=messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=max_tokens,
                tools=tools,
                **(extra or {}),
            )

        span_cm = (
            llm_span(f"llm.{op}", provider=self.provider.name, model=effective_model)
            if self.tracing
            else contextlib.nullcontext(None)
        )
        async with span_cm as span:
            emitted = 0
            usage: TokenUsage | None = None
            cost_usd = 0.0
            for attempt in range(1, attempts + 1):
                try:
                    stream = await _open()
                    async for event in stream:
                        emitted += 1
                        if isinstance(event, UsageDelta):
                            usage = event.usage
                            cost_usd = event.cost_usd
                            # Recorded the moment the provider reports usage, through the
                            # same CostTracker.record() that complete() calls — so a
                            # tagged streamed run lands under its tag, and an unpriced
                            # model still registers in `untracked_models`.
                            if record_cost and self.cost_tracker is not None:
                                self.cost_tracker.record(
                                    CompletionResult(
                                        content="",
                                        model=effective_model,
                                        provider=self.provider.name,
                                        usage=usage,
                                        cost_usd=cost_usd,
                                    ),
                                    tag=tag,
                                )
                        yield event
                    break
                except Exception as exc:
                    # Narrowed with an `is None` guard rather than folded into a
                    # `retryable` bool: mypy cannot carry a None-check through a
                    # variable, so the delay_for call below was unchecked.
                    if policy is None:
                        raise
                    retryable = (
                        emitted == 0 and attempt < attempts and isinstance(exc, policy.retry_on)
                    )
                    if not retryable:
                        raise
                    delay = policy.delay_for(attempt + 1)
                    if delay > 0:
                        await asyncio.sleep(delay)

            if span is not None:
                span.set_attribute("llm.stream_events", emitted)
                if usage is not None:
                    span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                    span.set_attribute("llm.completion_tokens", usage.completion_tokens)
                span.set_attribute("llm.cost_usd", cost_usd)

    async def run_agent_stream(
        self,
        prompt: str | list[ChatMessage],
        tools: ToolRegistry,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_steps: int = 6,
        system: str | None = None,
        tag: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming agent loop. Yields text deltas while the model is thinking,
        dispatches tool calls as they complete, and loops until a final text answer.

        ``tag`` is recorded on the CostTracker for every step of the loop, matching
        `run_agent`."""
        _require_registry(tools)
        messages = self._normalize(prompt, system=system)
        specs = tools.as_specs()
        self._require_tool_support(specs, streaming=True)
        effective_model = model or self.settings.model
        effective_temp = temperature if temperature is not None else self.settings.temperature

        for _ in range(max_steps):
            step_tool_calls = []
            step_text_parts: list[str] = []
            async for event in self._stream_layered(
                messages,
                model=effective_model,
                temperature=effective_temp,
                max_tokens=None,
                tools=specs,
                op="run_agent_stream",
                tag=tag,
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

    async def _run(
        self,
        coro_fn: Callable[[], Awaitable[CompletionResult]],
        *,
        op: str,
        model: str,
    ) -> CompletionResult:
        call: Callable[[], Awaitable[CompletionResult]] = coro_fn
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


def _extract_payload(result: CompletionResult, plan: SchemaPlan) -> str:
    """Pull the JSON body out of a completion, wherever that provider put it.

    The forced-tool path is the one case where the answer is not in ``content`` at all:
    the model's whole response is the tool call's ``arguments``. Everything else — the
    other native modes and the prompt path — returns text, which still goes through
    `_extract_json` because a native mode guarantees the *schema*, not the absence
    of a code fence around it.
    """
    if plan.native and plan.mode == "anthropic_tool":
        for call in result.tool_calls:
            if call.name == ANTHROPIC_EXTRACT_TOOL:
                return json.dumps(call.arguments)
        raise ValueError(
            f"Provider was required to call {ANTHROPIC_EXTRACT_TOOL!r} and did not. "
            f"Got {len(result.tool_calls)} tool call(s) and content {result.content[:200]!r}."
        )
    return _extract_json(result.content)


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
