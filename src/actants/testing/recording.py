"""Record a real agent run, then replay it offline.

An agent's behaviour is a function of model, prompt, and tools. Change one and the only
honest way to know what broke is to run the thing again — which costs money, needs a
network, and is not reproducible. A recording turns one real run into an artifact that
answers the question offline, in milliseconds, forever.

The format is JSONL: one header line, then one line per LLM exchange. Diffable in a PR,
greppable, and readable by anything that can read a file. What is recorded is the
*provider* boundary — the request that went on the wire and the completion that came
back — because that is the one place every path through actants passes through, and the
only place a replay can serve an answer without a network.

Record::

    recorder = RunRecorder("runs/booking.jsonl")
    agent = Agent(llm=LLM(provider=recorder.wrap(OllamaProvider()), model="qwen2.5:7b"),
                  tools=tools)
    result = await agent.run("book a flight to Berlin")
    recorder.close()

Replay, with no network at all::

    recording = Recording.load("runs/booking.jsonl")
    agent = Agent(llm=LLM(provider=ReplayProvider(recording), model="qwen2.5:7b"),
                  tools=tools)
    result = await agent.run("book a flight to Berlin")   # identical, offline

Tool calls are *not* served from the recording: the agent re-dispatches them against the
real registry, and `ReplayProvider` records what it saw so a replayed run can be
compared against the original trajectory. That is deliberate — replaying tool results
too would mean a change in a tool's own logic could never be caught, which is half of
what this exists to detect.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from actants.errors import (
    RecordingFormatError,
    RecordingMissError,
)
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
from actants.storage.jsonl import JsonlAppender

#: Version of the on-disk recording format, written into every file's header.
#:
#: Bumped whenever the shape of a line changes. A recording is a regression baseline
#: whose whole value is that it means the same thing tomorrow as it did when it was
#: recorded, so a mismatch is always an error — never a best-effort read. See
#: `load`.
FORMAT_VERSION = 1

#: How a replay decides which recorded exchange answers a request.
#:
#: ``"sequence"`` serves exchanges in the order they were recorded and is what makes a
#: replay against a *different* prompt or model possible at all — the request will not
#: match, and matching it is not the point. ``"request"`` looks the request up by its
#: exact content and refuses to serve a mismatch, which is what makes a determinism check
#: mean something.
MatchMode = Literal["sequence", "request"]

_MATCH_MODES: tuple[MatchMode, ...] = ("sequence", "request")


class RecordedRequest(BaseModel):
    """The request that went to the provider, in a form that survives a round-trip."""

    messages: list[ChatMessage] = Field(default_factory=list)
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[ToolSpec] | None = None
    #: Provider-specific passthrough (``seed``, ``top_p``, ...), JSON-coerced.
    extra: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """A stable identity for ``match="request"`` lookups.

        Canonical JSON of every field that can change the answer — the same principle as
        `CacheRequest`, computed here rather than reused
        because a recording must keep meaning the same thing when the cache key version
        moves for cache reasons.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class RecordedExchange(BaseModel):
    """One LLM round-trip: what was asked, what came back, and how long it took."""

    #: Position in the run, from 0. Ordering is the recording's spine — ``match="sequence"``
    #: reads it straight through.
    index: int
    request: RecordedRequest
    response: CompletionResult
    #: Wall-clock duration of the provider call, measured at record time. Kept separate
    #: from ``response.latency_ms``, which providers may or may not populate.
    latency_ms: float = 0.0
    #: True when the exchange was captured from ``stream_events`` rather than ``complete``.
    streamed: bool = False


class RecordingHeader(BaseModel):
    """The first line of every recording file.

    Exists so that a file written by an older actants fails loudly on load rather than
    being read with today's assumptions about what its fields mean.
    """

    kind: Literal["actants.recording"] = "actants.recording"
    format_version: int = FORMAT_VERSION
    #: The provider the run was recorded against, for the report a diff prints.
    provider: str = ""
    #: The model the run was recorded against.
    model: str = ""
    #: Free-form label — the case name, the git sha, whatever the caller wants.
    label: str | None = None
    created_at: float = Field(default_factory=time.time)


class Recording(BaseModel):
    """A recorded run: a header plus every LLM exchange, in order.

    Load one with `load`, serve it with `ReplayProvider`. A recording is
    immutable in spirit — nothing in actants mutates a loaded one — so the same object can
    seed several replays.
    """

    header: RecordingHeader = Field(default_factory=RecordingHeader)
    exchanges: list[RecordedExchange] = Field(default_factory=list)
    #: Where this was loaded from, for error messages. None for one built in memory.
    path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __len__(self) -> int:
        return len(self.exchanges)

    @property
    def total_cost_usd(self) -> float:
        """What the recorded run cost, as the provider reported it."""
        return sum(e.response.cost_usd for e in self.exchanges)

    @property
    def total_latency_ms(self) -> float:
        """Wall-clock time the recorded run spent waiting on the provider."""
        return sum(e.latency_ms for e in self.exchanges)

    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Every tool call the recorded model asked for, in order, as ``(name, args)``.

        This is the recorded *trajectory* — what a
        `ToolCalled` assertion is checked against.
        """
        return [(c.name, dict(c.arguments)) for e in self.exchanges for c in e.response.tool_calls]

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Read a recording from a JSONL file.

        Raises `RecordingFormatError` for a file this build cannot
        read: a wrong format version, a missing header, or a line that does not parse.
        Never a partial read — a recording is a regression baseline, and one that silently
        drops the exchanges it could not understand would report a passing replay of a run
        that never happened.
        """
        p = Path(path)
        if not p.exists():
            raise RecordingFormatError(
                f"No recording at {str(p)!r}. Record one first with "
                "RunRecorder(path).wrap(provider), or check the path."
            )
        lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise RecordingFormatError(
                f"Recording at {str(p)!r} is empty. It should start with a header line "
                f"({{'kind': 'actants.recording', 'format_version': {FORMAT_VERSION}}}) "
                "followed by one line per LLM exchange."
            )
        header = cls._parse_header(lines[0], p)
        exchanges: list[RecordedExchange] = []
        for lineno, line in enumerate(lines[1:], start=2):
            try:
                exchanges.append(RecordedExchange.model_validate_json(line))
            except Exception as exc:
                raise RecordingFormatError(
                    f"Recording at {str(p)!r} has an unreadable exchange on line {lineno}: "
                    f"{exc}. The file may have been truncated mid-write, or hand-edited."
                ) from exc
        return cls(header=header, exchanges=exchanges, path=p)

    @staticmethod
    def _parse_header(line: str, path: Path) -> RecordingHeader:
        """Read line 1, refusing anything this build cannot interpret."""
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordingFormatError(
                f"Recording at {str(path)!r} does not start with a JSON header line: {exc}."
            ) from exc
        if not isinstance(raw, dict) or raw.get("kind") != "actants.recording":
            raise RecordingFormatError(
                f"Recording at {str(path)!r} does not start with an actants recording "
                f"header (got {str(raw)[:120]!r}). The first line must be a "
                "{'kind': 'actants.recording', ...} object."
            )
        found = raw.get("format_version")
        if found != FORMAT_VERSION:
            raise RecordingFormatError(
                f"Recording at {str(path)!r} was written in format version {found!r}, and "
                f"this build of actants reads version {FORMAT_VERSION}. A recording is a "
                "regression baseline, so actants will not guess at an unfamiliar layout — "
                "a misread baseline reports a passing replay of a run that never "
                "happened. Re-record it with this version of actants, or check out the "
                "version that wrote it."
            )
        try:
            return RecordingHeader.model_validate(raw)
        except Exception as exc:
            raise RecordingFormatError(
                f"Recording at {str(path)!r} has a header this build cannot read: {exc}."
            ) from exc


class RunRecorder:
    """Captures every LLM exchange a run makes, to a JSONL file or to memory.

    Wrap any provider with `wrap` and use the result exactly as you would the
    original: it delegates every call through and writes down what happened. The run
    behaves identically — recording is observation, never interception.

    Example::

        recorder = RunRecorder("runs/case.jsonl", label="booking-v1")
        llm = LLM(provider=recorder.wrap(OllamaProvider()), model="qwen2.5:7b")
        await Agent(llm=llm, tools=tools).run("book a flight")
        recorder.close()

    ``path=None`` records to memory only, which is what a test that never wants a file
    wants; `recording` is available either way.
    """

    def __init__(self, path: str | Path | None = None, *, label: str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._header = RecordingHeader(label=label)
        self._exchanges: list[RecordedExchange] = []
        self._appender: JsonlAppender | None = None
        self._header_written = False

    @property
    def path(self) -> Path | None:
        """The file being written, or None for an in-memory recorder. **Read-only.**"""
        return self._path

    @property
    def recording(self) -> Recording:
        """The run captured so far. Safe to read mid-run."""
        return Recording(header=self._header, exchanges=list(self._exchanges), path=self._path)

    def wrap(self, provider: BaseLLMProvider) -> RecordingProvider:
        """Return ``provider`` with recording attached."""
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError(
                f"wrap() expects a BaseLLMProvider, got {type(provider).__name__!r}. "
                "Pass the provider itself, not the LLM: "
                "LLM(provider=recorder.wrap(OllamaProvider()))."
            )
        # Stamped from the first provider wrapped, so a loaded recording says what it was
        # recorded against without the caller having to repeat it.
        if not self._header.provider:
            self._header.provider = provider.name
        return RecordingProvider(provider, self)

    def add(self, exchange: RecordedExchange) -> None:
        """Append one exchange, writing it through to disk if this recorder has a path."""
        self._exchanges.append(exchange)
        if not self._header.model:
            self._header.model = exchange.request.model
        if self._path is None:
            return
        if self._appender is None:
            self._appender = JsonlAppender(self._path)
        if not self._header_written:
            # Written lazily so a recorder that never sees a call leaves no half-file, and
            # so the header can carry the model the run actually used.
            self._appender.write(self._header.model_dump(mode="json"))
            self._header_written = True
        self._appender.write(exchange.model_dump(mode="json"))

    def close(self) -> None:
        """Flush and close the underlying file. Idempotent."""
        if self._appender is not None:
            self._appender.close()
            self._appender = None

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._exchanges)

    def __repr__(self) -> str:
        where = str(self._path) if self._path is not None else "<memory>"
        return f"RunRecorder(path={where!r}, exchanges={len(self._exchanges)})"


class RecordingProvider(BaseLLMProvider):
    """A provider that delegates every call and writes down what happened.

    Built by `wrap`. Capability flags mirror the wrapped provider's, so
    wrapping cannot change what the client believes the provider can do.
    """

    def __init__(self, inner: BaseLLMProvider, recorder: RunRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self.name = inner.name
        self.native_schema_mode = inner.native_schema_mode

    @property
    def inner(self) -> BaseLLMProvider:
        """The provider being recorded. **Read-only.**"""
        return self._inner

    # Derived on access rather than copied at construction, for the reason
    # FallbackProvider derives its own: a provider that learns its capabilities from a
    # runtime handshake would otherwise be pinned to whatever it claimed at wrap time.
    @property
    def supports_tool_calls(self) -> bool:  # type: ignore[override]
        return self._inner.supports_tool_calls

    @property
    def supports_streaming_tools(self) -> bool:  # type: ignore[override]
        return self._inner.supports_streaming_tools

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
        started = time.perf_counter()
        result = await self._inner.complete(
            messages, model, temperature, max_tokens, tools=tools, **kwargs
        )
        elapsed = (time.perf_counter() - started) * 1000
        self._recorder.add(
            RecordedExchange(
                index=len(self._recorder),
                request=_build_request(messages, model, temperature, max_tokens, tools, kwargs),
                response=result,
                latency_ms=elapsed,
            )
        )
        return result

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
        """Pass events through, assembling the completion they add up to.

        A streamed exchange is recorded as the `CompletionResult` it produced, not
        as a list of deltas: the replay serves whole completions and re-derives deltas from
        them, so one recording drives both a streamed and a non-streamed replay.
        """
        started = time.perf_counter()
        text: list[str] = []
        calls = []
        usage = None
        cost = 0.0
        finish: FinishDelta | None = None
        async for event in self._inner.stream_events(
            messages, model, temperature, max_tokens, tools=tools, **kwargs
        ):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, ToolCallDelta):
                calls.append(event.tool_call)
            elif isinstance(event, UsageDelta):
                usage = event.usage
                cost = event.cost_usd
            elif isinstance(event, FinishDelta):
                finish = event
            yield event
        elapsed = (time.perf_counter() - started) * 1000
        result = CompletionResult(
            content="".join(text),
            model=model,
            provider=self._inner.name,
            usage=usage if usage is not None else TokenUsage(),
            cost_usd=cost,
            latency_ms=elapsed,
            finish_reason=finish.reason if finish is not None else "unknown",
            raw_finish_reason=finish.raw_reason if finish is not None else None,
            tool_calls=calls,
        )
        self._recorder.add(
            RecordedExchange(
                index=len(self._recorder),
                request=_build_request(messages, model, temperature, max_tokens, tools, kwargs),
                response=result,
                latency_ms=elapsed,
                streamed=True,
            )
        )

    async def health(self) -> bool:
        return await self._inner.health()

    def __repr__(self) -> str:
        return f"RecordingProvider({self._inner!r})"


class ReplayProvider(BaseLLMProvider):
    """Serves a `Recording` back in place of a real provider. **No network.**

    This is the piece that turns a recorded run into an offline regression suite: an agent
    built on one behaves exactly as it did when recorded, in microseconds, with no server
    running and no key set.

    Two matching modes, and the choice says what the replay is *for*:

    * ``match="sequence"`` (the default) hands back exchange 0, then 1, then 2. Use it to
      replay a recorded run against a **different prompt, model, or tool set** — the
      request will not match, and the point is to see what the rest of the system does
      with the same model answers.
    * ``match="request"`` looks each request up by content and raises
      `RecordingMissError` on a request that was never recorded.
      Use it for a **determinism check**: it proves the run asked the same questions, not
      merely the same number of them.

    Running past the end of the recording raises rather than inventing an answer::

        agent = Agent(llm=LLM(provider=ReplayProvider(recording), model="qwen2.5:7b"),
                      tools=tools)
        result = await agent.run("book a flight to Berlin")
    """

    name = "replay"

    def __init__(
        self,
        recording: Recording,
        *,
        match: MatchMode = "sequence",
        model: str | None = None,
    ) -> None:
        if not isinstance(recording, Recording):
            raise TypeError(
                f"ReplayProvider expects a Recording, got {type(recording).__name__!r}. "
                "Load one with Recording.load('runs/case.jsonl')."
            )
        if match not in _MATCH_MODES:
            raise ValueError(
                f"match must be one of {list(_MATCH_MODES)}, got {match!r}. "
                "'sequence' serves the recorded exchanges in order (use it to replay "
                "against a different prompt or model); 'request' looks each request up by "
                "content and refuses a mismatch (use it to check determinism)."
            )
        self._recording = recording
        self._match: MatchMode = match
        self._cursor = 0
        self._served: list[RecordedExchange] = []
        self._requests: list[RecordedRequest] = []
        # Reported as this provider's model so a replayed CompletionResult names what was
        # actually replayed rather than the literal string "replay".
        self._model = model
        self._by_fingerprint: dict[str, list[RecordedExchange]] = {}
        for exchange in recording.exchanges:
            self._by_fingerprint.setdefault(exchange.request.fingerprint(), []).append(exchange)

    # Declared True unconditionally: the recording already contains whatever tool calls the
    # real provider made, so refusing tools here would make every recorded tool-calling run
    # unreplayable. A recording of a tool-less run simply has no tool calls in it.
    supports_tool_calls = True
    supports_streaming_tools = True

    @property
    def recording(self) -> Recording:
        """The recording being served. **Read-only.**"""
        return self._recording

    @property
    def served(self) -> list[RecordedExchange]:
        """The exchanges served so far, in order. **Read-only view.**"""
        return list(self._served)

    @property
    def requests(self) -> list[RecordedRequest]:
        """The requests the replayed run actually made, in order.

        This is what makes a replay *comparable*: the recording says what the original run
        asked, and this says what the run under test asked. A diff of the two is the
        answer to "did my prompt change break anything".
        """
        return list(self._requests)

    @property
    def exhausted(self) -> bool:
        """True when every recorded exchange has been served."""
        return self._cursor >= len(self._recording.exchanges)

    def reset(self) -> None:
        """Rewind to the start, so one ReplayProvider can drive a second run."""
        self._cursor = 0
        self._served.clear()
        self._requests.clear()

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
        request = _build_request(messages, model, temperature, max_tokens, tools, kwargs)
        self._requests.append(request)
        exchange = self._next(request)
        self._served.append(exchange)
        # Copied: the caller owns what it gets back, and a caller that mutates a replayed
        # result must not corrupt the recording for the next replay of the same object.
        return exchange.response.model_copy(deep=True)

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
        """Re-derive a stream from the recorded completion.

        Text arrives as one delta rather than the original chunking: the chunk boundaries
        are a provider artifact that nothing downstream may depend on, and pretending to
        reproduce them would be a fiction the recording cannot actually support.
        """
        result = await self.complete(
            messages, model, temperature, max_tokens, tools=tools, **kwargs
        )
        if result.content:
            yield TextDelta(text=result.content)
        for call in result.tool_calls:
            yield ToolCallDelta(tool_call=call)
        yield UsageDelta(usage=result.usage, cost_usd=result.cost_usd)
        yield FinishDelta(reason=result.finish_reason, raw_reason=result.raw_finish_reason)

    def _next(self, request: RecordedRequest) -> RecordedExchange:
        """Pick the exchange that answers ``request``, or explain why none does."""
        if self._match == "request":
            candidates = self._by_fingerprint.get(request.fingerprint())
            if not candidates:
                raise RecordingMissError(
                    self._miss_message(request),
                    request_index=len(self._requests) - 1,
                )
            # Repeated identical requests are served in recorded order, so a run that asks
            # the same question twice gets both recorded answers rather than the first one
            # twice — which is exactly the case a retry or a loop produces.
            index = sum(1 for r in self._requests[:-1] if r.fingerprint() == request.fingerprint())
            if index >= len(candidates):
                raise RecordingMissError(
                    f"This run made the same request {index + 1} times, but the recording "
                    f"at {self._where()} has only {len(candidates)} recorded answer(s) for "
                    "it. The run under test is looping more than the recorded one did.",
                    request_index=len(self._requests) - 1,
                )
            return candidates[index]

        if self._cursor >= len(self._recording.exchanges):
            raise RecordingMissError(
                f"The replayed run asked for LLM call {self._cursor + 1}, but the "
                f"recording at {self._where()} has only "
                f"{len(self._recording.exchanges)}. The run under test takes more steps "
                "than the recorded one did — a prompt or tool change made the model loop "
                "longer. Re-record it, or raise the recorded run's max_steps.",
                request_index=len(self._requests) - 1,
            )
        exchange = self._recording.exchanges[self._cursor]
        self._cursor += 1
        return exchange

    def _miss_message(self, request: RecordedRequest) -> str:
        """Explain a request-mode miss in terms of what actually differs."""
        recorded_models = {e.request.model for e in self._recording.exchanges}
        detail = ""
        if request.model not in recorded_models:
            detail = (
                f" This run used model {request.model!r}; the recording was made against "
                f"{sorted(recorded_models)}. Replaying against a different model needs "
                "match='sequence' — request matching is for determinism checks against "
                "the same model."
            )
        return (
            f"No recorded exchange matches LLM call {len(self._requests)} of this run, and "
            f"the recording at {self._where()} is being replayed with match='request', "
            "which will not substitute a different one." + detail
        )

    def _where(self) -> str:
        return str(self._recording.path) if self._recording.path is not None else "<memory>"

    async def health(self) -> bool:
        """Always reachable: there is nothing to reach."""
        return True

    def __repr__(self) -> str:
        return (
            f"ReplayProvider(exchanges={len(self._recording.exchanges)}, "
            f"match={self._match!r}, served={len(self._served)})"
        )


def _build_request(
    messages: Sequence[ChatMessage],
    model: str,
    temperature: float,
    max_tokens: int | None,
    tools: list[ToolSpec] | None,
    extra: dict[str, object],
) -> RecordedRequest:
    """Describe one provider call in the recording's own vocabulary."""
    return RecordedRequest(
        messages=[m.model_copy(deep=True) for m in messages],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[t.model_copy(deep=True) for t in tools] if tools is not None else None,
        # default=str so a provider kwarg holding a non-JSON object cannot make a run
        # unrecordable — the recording is observation and must never break the run.
        extra=json.loads(json.dumps(extra, default=str)) if extra else {},
    )


def iter_exchanges(path: str | Path) -> Iterator[RecordedExchange]:
    """Stream a recording's exchanges without holding the whole file in memory.

    For the case `load` is wrong for: a long run being scanned for one
    thing. Validates the header first, so a bad file still fails on the first line.
    """
    return _iter_exchanges(Path(path))


def _iter_exchanges(path: Path) -> Iterator[RecordedExchange]:
    with path.open("r", encoding="utf-8") as fh:
        lines = (line for line in fh if line.strip())
        try:
            first = next(lines)
        except StopIteration:
            raise RecordingFormatError(f"Recording at {str(path)!r} is empty.") from None
        Recording._parse_header(first, path)
        for lineno, line in enumerate(lines, start=2):
            try:
                yield RecordedExchange.model_validate_json(line)
            except Exception as exc:
                raise RecordingFormatError(
                    f"Recording at {str(path)!r} has an unreadable exchange on line "
                    f"{lineno}: {exc}."
                ) from exc


__all__ = [
    "FORMAT_VERSION",
    "MatchMode",
    "RecordedExchange",
    "RecordedRequest",
    "Recording",
    "RecordingHeader",
    "RecordingProvider",
    "ReplayProvider",
    "RunRecorder",
    "iter_exchanges",
]
