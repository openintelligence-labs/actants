"""Record and replay: a recorded run must be reproducible offline, forever.

The contract these pin down:

1. record -> replay reproduces the original run *exactly* (content, tool calls, order)
2. replay touches no network, and works with the real provider unreachable
3. a recording replays against a *different* model
4. a format-version mismatch fails loudly rather than being read with today's assumptions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from actants.agents.agent import Agent
from actants.errors import ActantsError, RecordingFormatError, RecordingMissError
from actants.llm.base import (
    ChatMessage,
    CompletionResult,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.client import LLM
from actants.testing import (
    FakeLLMProvider,
    Recording,
    ReplayProvider,
    RunRecorder,
    fake_completion,
    fake_tool_call_completion,
    iter_exchanges,
)
from actants.testing.recording import FORMAT_VERSION
from actants.tools.registry import ToolRegistry


def _registry(log: list[tuple[str, dict[str, Any]]] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        if log is not None:
            log.append(("add", {"a": a, "b": b}))
        return a + b

    registry.register_function("add", "Add two integers", add)
    return registry


def _tool_run_provider() -> FakeLLMProvider:
    """A scripted two-step run: one tool call, then a final answer."""
    return FakeLLMProvider(
        [
            fake_tool_call_completion("add", {"a": 2, "b": 3}, call_id="t1"),
            fake_completion("The answer is 5"),
        ]
    )


# ---------------------------------------------------------------------------
# 1. record -> replay reproduces the original run exactly
# ---------------------------------------------------------------------------


async def test_replay_reproduces_the_original_run_exactly(tmp_path: Path) -> None:
    """The headline guarantee: same content, same tool calls, same order."""
    path = tmp_path / "run.jsonl"
    recorder = RunRecorder(path)
    original_tools = _registry()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=original_tools,
    )
    original = await agent.run("what is 2+3?")
    recorder.close()

    recording = Recording.load(path)
    replayed_agent = Agent(
        llm=LLM(provider=ReplayProvider(recording), model="fake", tracing=False),
        tools=_registry(),
    )
    replayed = await replayed_agent.run("what is 2+3?")

    assert replayed.content == original.content == "The answer is 5"
    assert len(replayed.steps) == len(original.steps) == 2
    assert [c.name for s in replayed.steps for c in s.tool_calls] == ["add"]
    assert [(c.name, c.arguments) for s in replayed.steps for c in s.tool_calls] == [
        (c.name, c.arguments) for s in original.steps for c in s.tool_calls
    ]
    # The messages the run built up, including the tool results, must agree too.
    assert [(m.role, m.content) for m in replayed.messages] == [
        (m.role, m.content) for m in original.messages
    ]


async def test_recording_captures_every_exchange_in_order(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = RunRecorder(path, label="case-1")
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")
    recorder.close()

    recording = Recording.load(path)
    assert len(recording) == 2
    assert [e.index for e in recording.exchanges] == [0, 1]
    assert recording.header.label == "case-1"
    assert recording.header.provider == "fake"
    assert recording.header.model == "fake"
    assert recording.tool_calls() == [("add", {"a": 2, "b": 3})]


async def test_recording_is_plain_jsonl_one_line_per_exchange(tmp_path: Path) -> None:
    """Diffable and greppable is the point of the format; assert it really is."""
    path = tmp_path / "run.jsonl"
    recorder = RunRecorder(path)
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")
    recorder.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "one header line plus one line per exchange"
    header = json.loads(lines[0])
    assert header["kind"] == "actants.recording"
    assert header["format_version"] == FORMAT_VERSION
    for line in lines[1:]:
        assert "index" in json.loads(line)


async def test_recording_does_not_change_the_run(tmp_path: Path) -> None:
    """Recording is observation. A recorded run must equal an unrecorded one."""
    plain = Agent(
        llm=LLM(provider=_tool_run_provider(), model="fake", tracing=False), tools=_registry()
    )
    recorded = Agent(
        llm=LLM(
            provider=RunRecorder(tmp_path / "r.jsonl").wrap(_tool_run_provider()),
            model="fake",
            tracing=False,
        ),
        tools=_registry(),
    )
    a = await plain.run("what is 2+3?")
    b = await recorded.run("what is 2+3?")
    assert a.content == b.content
    assert len(a.steps) == len(b.steps)


async def test_in_memory_recorder_needs_no_file() -> None:
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")
    assert recorder.path is None
    assert len(recorder.recording) == 2

    replayed = Agent(
        llm=LLM(provider=ReplayProvider(recorder.recording), model="fake", tracing=False),
        tools=_registry(),
    )
    assert (await replayed.run("what is 2+3?")).content == "The answer is 5"


async def test_replay_re_dispatches_tools_rather_than_replaying_their_results() -> None:
    """A tool's own logic must stay under test; only the LLM is served from the tape.

    If tool results were replayed too, a bug introduced in a tool would replay green.
    """
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")

    calls: list[tuple[str, dict[str, Any]]] = []
    replayed = Agent(
        llm=LLM(provider=ReplayProvider(recorder.recording), model="fake", tracing=False),
        tools=_registry(calls),
    )
    await replayed.run("what is 2+3?")
    assert calls == [("add", {"a": 2, "b": 3})], "the real tool must run during a replay"


# ---------------------------------------------------------------------------
# 2. replay with zero network
# ---------------------------------------------------------------------------


async def test_replay_makes_no_network_call_at_all(tmp_path: Path, monkeypatch) -> None:
    """The real prize: a recorded run becomes a fast offline regression test.

    Every httpx entry point is poisoned, so a replay that touched the network in any
    way — even through a layer this test does not know about — fails loudly.
    """
    path = tmp_path / "run.jsonl"
    recorder = RunRecorder(path)
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    original = await agent.run("what is 2+3?")
    recorder.close()

    import httpx

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("replay attempted a network request")

    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)

    recording = Recording.load(path)
    offline = Agent(
        llm=LLM(provider=ReplayProvider(recording), model="fake", tracing=False),
        tools=_registry(),
    )
    replayed = await offline.run("what is 2+3?")
    assert replayed.content == original.content


async def test_replay_provider_health_is_true_without_a_server() -> None:
    provider = ReplayProvider(Recording())
    assert await provider.health() is True


async def test_replay_never_touches_the_provider_it_replaces() -> None:
    """A replay is built from the recording alone; the original provider is irrelevant."""
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")

    # ExplodingProvider is deliberately *not* wired in — the point is that a ReplayProvider
    # needs nothing but the tape. Constructing it proves the API takes no provider at all.
    tape = recorder.recording
    replay = ReplayProvider(tape)
    assert replay.recording is tape
    result = await replay.complete([ChatMessage(role="user", content="anything")], "fake")
    assert result.tool_calls[0].name == "add"


# ---------------------------------------------------------------------------
# 3. replay against a different model
# ---------------------------------------------------------------------------


async def test_replay_against_a_different_model_serves_the_recorded_answers() -> None:
    """Sequence matching is what makes 'what if I swap the model' answerable offline."""
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="gpt-4o", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")

    replay = ReplayProvider(recorder.recording, match="sequence")
    swapped = Agent(
        llm=LLM(provider=replay, model="claude-sonnet-5", tracing=False), tools=_registry()
    )
    result = await swapped.run("what is 2+3?")

    assert result.content == "The answer is 5"
    # ...and the replay recorded what the *new* configuration asked for, which is the
    # thing a caller diffs against the recording.
    assert [r.model for r in replay.requests] == ["claude-sonnet-5", "claude-sonnet-5"]
    assert [e.request.model for e in recorder.recording.exchanges] == ["gpt-4o", "gpt-4o"]


async def test_replay_against_a_different_prompt_still_serves_in_order() -> None:
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")

    replay = ReplayProvider(recorder.recording)
    other = Agent(llm=LLM(provider=replay, model="fake", tracing=False), tools=_registry())
    result = await other.run("completely different question")
    assert result.content == "The answer is 5"
    assert "completely different question" in replay.requests[0].messages[-1].content


async def test_request_matching_refuses_a_different_model() -> None:
    """A determinism check must not silently pass when the model changed under it."""
    recorder = RunRecorder()
    llm = LLM(provider=recorder.wrap(FakeLLMProvider([fake_completion("hi")])), model="gpt-4o")
    await llm.complete("hello", use_cache=False)

    replay = ReplayProvider(recorder.recording, match="request")
    swapped = LLM(provider=replay, model="claude-sonnet-5", tracing=False)
    with pytest.raises(RecordingMissError) as exc:
        await swapped.complete("hello", use_cache=False)
    message = str(exc.value)
    assert "claude-sonnet-5" in message
    assert "match='sequence'" in message


async def test_request_matching_serves_an_identical_request() -> None:
    """The determinism check must pass when nothing changed — otherwise it is useless."""
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("hi")])),
        model="fake",
        tracing=False,
    )
    await llm.complete("hello", use_cache=False)

    replay = ReplayProvider(recorder.recording, match="request")
    same = LLM(provider=replay, model="fake", tracing=False)
    assert (await same.complete("hello", use_cache=False)).content == "hi"


async def test_request_matching_serves_repeated_requests_in_recorded_order() -> None:
    """A loop that asks the same question twice must get both recorded answers."""
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(
            FakeLLMProvider([fake_completion("first"), fake_completion("second")])
        ),
        model="fake",
        tracing=False,
    )
    await llm.complete("same", use_cache=False)
    await llm.complete("same", use_cache=False)

    replay = ReplayProvider(recorder.recording, match="request")
    same = LLM(provider=replay, model="fake", tracing=False)
    assert (await same.complete("same", use_cache=False)).content == "first"
    assert (await same.complete("same", use_cache=False)).content == "second"

    with pytest.raises(RecordingMissError, match="looping more"):
        await same.complete("same", use_cache=False)


async def test_running_past_the_end_of_a_recording_raises() -> None:
    """A run that takes more steps than the recording must not get an invented answer."""
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("only one")])),
        model="fake",
        tracing=False,
    )
    await llm.complete("hi", use_cache=False)

    replay = ReplayProvider(recorder.recording)
    replayed = LLM(provider=replay, model="fake", tracing=False)
    await replayed.complete("hi", use_cache=False)
    with pytest.raises(RecordingMissError) as exc:
        await replayed.complete("hi", use_cache=False)
    assert exc.value.request_index == 1
    assert "more steps" in str(exc.value)


async def test_replay_can_be_reset_and_driven_twice() -> None:
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("once")])),
        model="fake",
        tracing=False,
    )
    await llm.complete("hi", use_cache=False)

    replay = ReplayProvider(recorder.recording)
    replayed = LLM(provider=replay, model="fake", tracing=False)
    assert (await replayed.complete("hi", use_cache=False)).content == "once"
    assert replay.exhausted
    replay.reset()
    assert not replay.exhausted
    assert (await replayed.complete("hi", use_cache=False)).content == "once"


# ---------------------------------------------------------------------------
# 4. format version mismatch fails loudly
# ---------------------------------------------------------------------------


def _write(path: Path, header: dict[str, Any], *rows: dict[str, Any]) -> Path:
    lines = [json.dumps(header), *(json.dumps(r) for r in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_future_format_version_is_refused(tmp_path: Path) -> None:
    """A recording is a baseline; reading one with the wrong assumptions is worse than
    refusing to read it, because a misread baseline reports a passing replay."""
    path = _write(
        tmp_path / "future.jsonl",
        {"kind": "actants.recording", "format_version": FORMAT_VERSION + 1},
    )
    with pytest.raises(RecordingFormatError) as exc:
        Recording.load(path)
    message = str(exc.value)
    assert str(FORMAT_VERSION + 1) in message
    assert str(FORMAT_VERSION) in message
    assert "Re-record" in message


def test_a_past_format_version_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "old.jsonl", {"kind": "actants.recording", "format_version": FORMAT_VERSION - 1}
    )
    with pytest.raises(RecordingFormatError):
        Recording.load(path)


def test_a_file_with_no_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "headless.jsonl"
    path.write_text(json.dumps({"index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(RecordingFormatError, match="actants recording"):
        Recording.load(path)


def test_a_non_json_first_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "garbage.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(RecordingFormatError, match="JSON header"):
        Recording.load(path)


def test_a_truncated_exchange_is_refused_not_skipped(tmp_path: Path) -> None:
    """Never a partial read — dropping an unreadable line would silently shorten a run."""
    path = tmp_path / "torn.jsonl"
    path.write_text(
        json.dumps({"kind": "actants.recording", "format_version": FORMAT_VERSION})
        + "\n"
        + '{"index": 0, "request": {"mo\n',
        encoding="utf-8",
    )
    with pytest.raises(RecordingFormatError, match="line 2"):
        Recording.load(path)


def test_an_empty_recording_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RecordingFormatError, match="empty"):
        Recording.load(path)


def test_a_missing_recording_names_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(RecordingFormatError, match="RunRecorder"):
        Recording.load(tmp_path / "nope.jsonl")


def test_recording_errors_are_catchable_as_actants_errors(tmp_path: Path) -> None:
    """`except ActantsError` must stay exhaustive over the new errors too."""
    with pytest.raises(ActantsError):
        Recording.load(tmp_path / "nope.jsonl")


def test_recording_format_error_is_also_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Recording.load(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Streaming, iteration, and the wrapper's contracts
# ---------------------------------------------------------------------------


async def test_a_streamed_run_records_and_replays(tmp_path: Path) -> None:
    """One recording drives both a streamed and a non-streamed replay."""
    path = tmp_path / "stream.jsonl"
    recorder = RunRecorder(path)
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("streamed answer")])),
        model="fake",
        tracing=False,
    )
    chunks = [c async for c in llm.stream("hi")]
    recorder.close()
    assert "".join(chunks) == "streamed answer"

    recording = Recording.load(path)
    assert recording.exchanges[0].streamed is True
    assert recording.exchanges[0].response.content == "streamed answer"

    replay = LLM(provider=ReplayProvider(recording), model="fake", tracing=False)
    # Replayed as a stream...
    assert "".join([c async for c in replay.stream("hi")]) == "streamed answer"
    # ...and, from the same tape, as a completion.
    replay2 = LLM(provider=ReplayProvider(recording), model="fake", tracing=False)
    assert (await replay2.complete("hi", use_cache=False)).content == "streamed answer"


async def test_a_replayed_stream_yields_tool_calls_and_usage() -> None:
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(
            FakeLLMProvider([fake_tool_call_completion("add", {"a": 1, "b": 2})])
        ),
        model="fake",
        tracing=False,
    )
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    async for _ in llm.stream_events("hi", tools=[spec]):
        pass

    replay = LLM(provider=ReplayProvider(recorder.recording), model="fake", tracing=False)
    events = [e async for e in replay.stream_events("hi", tools=[spec])]
    assert any(isinstance(e, ToolCallDelta) and e.tool_call.name == "add" for e in events)
    assert any(isinstance(e, UsageDelta) for e in events)


async def test_recording_provider_mirrors_the_wrapped_capability_flags() -> None:
    """Wrapping must not change what the client believes the provider can do."""
    inner = FakeLLMProvider()
    wrapped = RunRecorder().wrap(inner)
    assert wrapped.supports_tool_calls is True
    inner.supports_tool_calls = False
    assert wrapped.supports_tool_calls is False, "derived on access, not snapshotted"
    assert wrapped.name == inner.name
    assert wrapped.inner is inner


def test_wrap_rejects_something_that_is_not_a_provider() -> None:
    with pytest.raises(TypeError, match="BaseLLMProvider"):
        RunRecorder().wrap(LLM(provider=FakeLLMProvider()))  # type: ignore[arg-type]


def test_replay_provider_rejects_a_non_recording() -> None:
    with pytest.raises(TypeError, match="Recording"):
        ReplayProvider({"exchanges": []})  # type: ignore[arg-type]


def test_replay_provider_rejects_an_unknown_match_mode() -> None:
    with pytest.raises(ValueError) as exc:
        ReplayProvider(Recording(), match="fuzzy")  # type: ignore[arg-type]
    assert "sequence" in str(exc.value) and "request" in str(exc.value)


async def test_iter_exchanges_streams_without_loading_everything(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = RunRecorder(path)
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")
    recorder.close()

    assert [e.index for e in iter_exchanges(path)] == [0, 1]


def test_iter_exchanges_validates_the_header_too(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.jsonl", {"kind": "something-else"})
    with pytest.raises(RecordingFormatError):
        list(iter_exchanges(path))


async def test_recording_reports_what_the_run_cost_and_took() -> None:
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(
            FakeLLMProvider(
                [
                    CompletionResult(content="a", model="gpt-4o", provider="fake", cost_usd=0.01),
                    CompletionResult(content="b", model="gpt-4o", provider="fake", cost_usd=0.02),
                ]
            )
        ),
        model="gpt-4o",
        tracing=False,
    )
    await llm.complete("one", use_cache=False)
    await llm.complete("two", use_cache=False)

    recording = recorder.recording
    assert recording.total_cost_usd == pytest.approx(0.03)
    assert recording.total_latency_ms >= 0.0


async def test_a_recorder_that_saw_nothing_writes_no_file(tmp_path: Path) -> None:
    """No half-file for a run that never made a call."""
    path = tmp_path / "unused.jsonl"
    recorder = RunRecorder(path)
    recorder.wrap(FakeLLMProvider())
    recorder.close()
    assert not path.exists()


async def test_recorder_works_as_a_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "ctx.jsonl"
    with RunRecorder(path) as recorder:
        llm = LLM(
            provider=recorder.wrap(FakeLLMProvider([fake_completion("done")])),
            model="fake",
            tracing=False,
        )
        await llm.complete("hi", use_cache=False)
    assert len(Recording.load(path)) == 1


async def test_replay_result_is_copied_so_a_caller_cannot_corrupt_the_tape() -> None:
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("original")])),
        model="fake",
        tracing=False,
    )
    await llm.complete("hi", use_cache=False)

    replay = ReplayProvider(recorder.recording)
    first = await replay.complete([ChatMessage(role="user", content="hi")], "fake")
    first.content = "mutated"
    replay.reset()
    second = await replay.complete([ChatMessage(role="user", content="hi")], "fake")
    assert second.content == "original"


async def test_passthrough_kwargs_are_recorded(tmp_path: Path) -> None:
    """Anything that changes the answer must appear in the recorded request."""
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("seeded")])),
        model="fake",
        tracing=False,
    )
    await llm.complete("hi", use_cache=False, seed=42)
    assert recorder.recording.exchanges[0].request.extra == {"seed": 42}


async def test_recorded_requests_carry_the_tool_specs(tmp_path: Path) -> None:
    recorder = RunRecorder()
    agent = Agent(
        llm=LLM(provider=recorder.wrap(_tool_run_provider()), model="fake", tracing=False),
        tools=_registry(),
    )
    await agent.run("what is 2+3?")
    tools = recorder.recording.exchanges[0].request.tools
    assert tools is not None
    assert [t.name for t in tools] == ["add"]


def test_recorder_repr_says_where_and_how_many(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "x.jsonl")
    assert "x.jsonl" in repr(recorder)
    assert "exchanges=0" in repr(recorder)
    assert "<memory>" in repr(RunRecorder())


async def test_replay_provider_repr_reports_progress() -> None:
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("x")])), model="fake", tracing=False
    )
    await llm.complete("hi", use_cache=False)
    replay = ReplayProvider(recorder.recording)
    assert "exchanges=1" in repr(replay)
    assert "served=0" in repr(replay)


async def test_text_deltas_survive_a_replayed_stream() -> None:
    """Chunk boundaries are a provider artifact; the text itself must be exact."""
    recorder = RunRecorder()
    llm = LLM(
        provider=recorder.wrap(FakeLLMProvider([fake_completion("hello world")])),
        model="fake",
        tracing=False,
    )
    async for _ in llm.stream("hi"):
        pass
    replay = LLM(provider=ReplayProvider(recorder.recording), model="fake", tracing=False)
    events = [e async for e in replay.stream_events("hi")]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "hello world"
