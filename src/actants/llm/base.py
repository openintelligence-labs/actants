from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from actants.llm.finish_reason import FinishReason
from actants.llm.structured import NativeSchemaMode

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResult(BaseModel):
    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    #: Why generation stopped, normalized across providers — see
    #: `FinishReason`. Safe to branch on: every provider
    #: maps onto the same six values, and an unrecognized or absent provider value
    #: becomes ``"unknown"`` rather than leaking through or raising.
    finish_reason: FinishReason = "unknown"
    #: The provider's own stop-reason string, exactly as it came off the wire
    #: (``"end_turn"``, ``"MAX_TOKENS"``, ``"tool_calls"``, ...), or ``None`` if the
    #: provider reported none. Nothing is lost by normalization; use this for logging or
    #: a provider-specific workaround, and `finish_reason` for control flow.
    raw_finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """Provider-agnostic tool description passed to an LLM for function calling."""

    name: str
    description: str
    #: JSON Schema describing the tool's arguments. Keys are always strings, so this is
    #: ``dict[str, Any]`` rather than a bare ``dict`` — under ``mypy --strict`` the
    #: latter degrades to ``dict[Any, Any]`` and loses the key type for consumers.
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallDelta(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class UsageDelta(BaseModel):
    type: Literal["usage"] = "usage"
    usage: TokenUsage
    cost_usd: float = 0.0


class FinishDelta(BaseModel):
    """The terminal event of a stream, carrying why generation stopped.

    ``reason`` is normalized the same way as
    `CompletionResult.finish_reason`, so a streamed run and a completed one can be
    branched on identically; ``raw_reason`` preserves the provider's own string.

    A provider may construct this with either — ``FinishDelta(reason="stop")`` in a
    hand-written provider, or via `from_provider` to normalize a raw wire value.
    """

    type: Literal["finish"] = "finish"
    reason: FinishReason = "unknown"
    #: The provider's own stop-reason string, verbatim. See
    #: `CompletionResult.raw_finish_reason`.
    raw_reason: str | None = None

    @classmethod
    def from_provider(cls, provider: str, raw: str | None) -> FinishDelta:
        """Build a ``FinishDelta`` from a provider's raw stop-reason string.

        This is what every built-in provider uses, so normalization happens in exactly
        one place per provider and the raw value is never dropped on the way.
        """
        from actants.llm.finish_reason import normalize_finish_reason

        return cls(reason=normalize_finish_reason(provider, raw), raw_reason=raw)


StreamEvent = TextDelta | ToolCallDelta | UsageDelta | FinishDelta


class BaseLLMProvider(ABC):
    """The contract every LLM provider implements.

    To write a provider, implement exactly three methods: `complete`,
    `stream_events`, and `health`. That is the whole surface.

    **Streaming has one primitive: ``stream_events``.** It yields typed
    `StreamEvent` objects, which is a superset of what plain text streaming can
    express — text deltas, tool calls, token usage, and the finish reason.
    `stream` is a *provided helper* that filters those events down to text; it is
    not an extension point, and overriding it has no effect on what
    `stream` yields, because the client also goes through
    ``stream_events``. Implement ``stream_events`` and both work.

    Capability flags are declared as plain class attributes::

        class MyProvider(BaseLLMProvider):
            name = "my-provider"
            supports_tool_calls = True
            supports_streaming_tools = True

    They may also be set per instance (a provider that learns what it can do from a
    runtime handshake), or replaced by a ``property`` in a subclass that derives them
    from something else — see `FallbackProvider`,
    which computes both from the providers in its chain on every access.

    Example::

        class EchoProvider(BaseLLMProvider):
            name = "echo"

            async def complete(self, messages, model, temperature=0.7,
                               max_tokens=None, *, tools=None, **kwargs):
                return CompletionResult(
                    content=messages[-1].content, model=model, provider=self.name
                )

            async def stream_events(self, messages, model, temperature=0.7,
                                    max_tokens=None, *, tools=None, **kwargs):
                yield TextDelta(text=messages[-1].content)
                yield FinishDelta(reason="stop")

            async def health(self) -> bool:
                return True
    """

    name: str

    #: Whether this provider can be given tool definitions.
    #: `LLM` refuses to pass tools to a provider that
    #: declares ``False``, because the specs would otherwise be dropped on the way to
    #: the wire and the model would answer as if no tools existed.
    supports_tool_calls: bool = False

    #: Whether this provider emits `ToolCallDelta` events while streaming.
    supports_streaming_tools: bool = False

    #: How this provider constrains output to a JSON Schema on the wire, if it can.
    #: :meth:`~actants.llm.client.LLM.extract` reads this to build the right request —
    #: ``response_format`` for OpenAI, a forced tool call for Anthropic, and so on — and
    #: falls back to describing the schema in a system prompt when it is ``"none"``.
    #:
    #: Declared here rather than discovered by ``isinstance`` so that a third-party
    #: provider gets the native path by setting one attribute, exactly as
    #: ``supports_tool_calls`` works. A provider whose endpoint speaks the OpenAI wire
    #: format but does *not* implement ``json_schema`` must leave this at ``"none"``:
    #: claiming it sends a request body that provider rejects.
    native_schema_mode: NativeSchemaMode = "none"

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        """Run a chat completion and return the result."""

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed events: text deltas, tool calls, usage, and finish.

        **This is the streaming primitive.** Override it as an ``async def`` generator
        that yields `TextDelta` for content, `ToolCallDelta` for each tool
        call, `UsageDelta` once token counts are known, and `FinishDelta`
        last. A provider that cannot stream tool calls simply never yields a
        ``ToolCallDelta`` and leaves ``supports_streaming_tools`` at ``False``.

        Not abstract, so a completion-only provider stays usable for
        `complete` without writing a streaming stub. The
        default raises `NotImplementedError` when something actually tries to
        stream.
        """
        raise NotImplementedError(
            f"Provider {type(self).__name__!r} does not implement streaming. "
            "Override `stream_events` as an async generator yielding TextDelta(...) for "
            "each content chunk and FinishDelta(reason=...) at the end; `stream` is then "
            "derived from it automatically. To use this provider without streaming, call "
            "llm.complete(...) instead of llm.stream(...) / llm.stream_events(...)."
        )
        yield  # pragma: no cover - unreachable; makes this an async generator

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Yield just the text content of `stream_events`.

        Provided by this base class — **do not override it.** Every actants code path
        that streams goes through ``stream_events``, so an override here would be
        bypassed and the two would silently disagree. Put your implementation in
        ``stream_events``; this helper picks the text out of it.
        """
        async for event in self.stream_events(
            messages,
            model,
            temperature,
            max_tokens,
            tools=None,
            **kwargs,
        ):
            if isinstance(event, TextDelta) and event.text:
                yield event.text

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject a subclass that overrides ``stream`` instead of ``stream_events``.

        Before ``stream_events`` existed, ``stream`` was the primitive; a provider
        written against that older shape still imports and constructs fine, but every
        actants entry point now reads ``stream_events``, so its override would never run
        and it would stream nothing. Failing at class-definition time turns a silent
        wrong answer into an error that names the fix.
        """
        super().__init_subclass__(**kwargs)
        if "stream" in cls.__dict__ and "stream_events" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides `stream` but not `stream_events`. "
                "`stream_events` is the streaming primitive in actants — every code "
                "path (LLM.stream, LLM.stream_events, Agent.stream) reads it — and "
                "`stream` is a helper the base class derives from it, so this override "
                "would never be called and the provider would appear to stream nothing. "
                f"Fix: rename {cls.__name__}.stream to `stream_events` and yield typed "
                "events (TextDelta(text=...) for content, then FinishDelta(reason=...)) "
                "instead of plain strings. The base class will then derive `stream` for "
                "you."
            )

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is reachable."""
