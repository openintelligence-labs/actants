"""Regression tests for API/protocol defects found by adversarial review.

Each defect changes a public interface, so each is fixed before 1.0 rather than after.
Every test here fails before its fix.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from typing import Any

import pytest

from actants.agents.agent import Agent
from actants.agents.events import (
    AgentRunCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)
from actants.agents.memory import ConversationMemory
from actants.cache.memory import InMemoryCache, make_key
from actants.cache.request import KEY_VERSION, CacheRequest
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolSpec,
    UsageDelta,
)
from actants.llm.client import LLM
from actants.llm.errors import ToolCallsNotSupportedError
from actants.policies.retry import RetryPolicy
from actants.testing import FakeLLMProvider, fake_completion, fake_tool_call_completion
from actants.tools.registry import ToolRegistry


def _result(text: str = "hi", model: str = "m") -> CompletionResult:
    return CompletionResult(
        content=text,
        model=model,
        provider="fake",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _msgs(text: str = "hello") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


# ---------------------------------------------------------------------------
# 1. Cache request identity
# ---------------------------------------------------------------------------


def test_cache_request_key_distinguishes_max_tokens() -> None:
    """The headline defect: same prompt, different output cap, different answer."""
    a = CacheRequest(messages=_msgs(), model="m", temperature=0.5, max_tokens=16)
    b = CacheRequest(messages=_msgs(), model="m", temperature=0.5, max_tokens=4096)
    assert a.key() != b.key()


def test_cache_request_scope_hash_distinguishes_max_tokens() -> None:
    """Semantic backends discriminate on the scope hash, so it must cover max_tokens.

    This is the exact collision the SqliteVecCache protocol allowed: message content is
    matched by embedding distance, so if max_tokens is not in the scope hash there is
    nothing left to tell the two requests apart.
    """
    a = CacheRequest(messages=_msgs(), model="m", temperature=0.5, max_tokens=16)
    b = CacheRequest(messages=_msgs(), model="m", temperature=0.5, max_tokens=4096)
    assert a.scope_hash() != b.scope_hash()


@pytest.mark.parametrize(
    ("field", "left", "right"),
    [
        ("provider", "ollama", "openai"),
        ("model", "llama3.2", "gpt-4o"),
        ("temperature", 0.0, 1.0),
        ("max_tokens", 16, 4096),
        ("response_format", None, {"type": "json_object"}),
    ],
)
def test_cache_request_scope_hash_covers_every_answer_changing_field(
    field: str, left: Any, right: Any
) -> None:
    base: dict[str, Any] = {"messages": _msgs(), "model": "m", "temperature": 0.5}
    a = CacheRequest(**{**base, field: left})
    b = CacheRequest(**{**base, field: right})
    assert a.scope_hash() != b.scope_hash(), f"{field} does not affect the scope hash"


def test_cache_request_scope_hash_distinguishes_tools() -> None:
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    a = CacheRequest(messages=_msgs(), model="m", temperature=0.5)
    b = CacheRequest(messages=_msgs(), model="m", temperature=0.5, tools=[spec])
    assert a.scope_hash() != b.scope_hash()


def test_cache_request_scope_hash_distinguishes_conversation_shape() -> None:
    """Message *count and roles* are not carried by the embedding, so they go in scope.

    Otherwise a 1-message and a 3-message conversation that flatten to similar text
    could match each other.
    """
    one = CacheRequest(messages=_msgs(), model="m", temperature=0.5)
    three = CacheRequest(
        messages=[
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
        ],
        model="m",
        temperature=0.5,
    )
    assert one.scope_hash() != three.scope_hash()


def test_cache_request_scope_hash_ignores_message_content() -> None:
    """Content is matched semantically, so it must NOT be in the scope hash."""
    a = CacheRequest(messages=_msgs("what is the capital of france"), model="m", temperature=0.5)
    b = CacheRequest(messages=_msgs("what's france's capital city"), model="m", temperature=0.5)
    assert a.scope_hash() == b.scope_hash()
    assert a.key() != b.key()


def test_cache_request_extra_is_part_of_the_key() -> None:
    a = CacheRequest(messages=_msgs(), model="m", temperature=0.5, extra={"seed": 1})
    b = CacheRequest(messages=_msgs(), model="m", temperature=0.5, extra={"seed": 2})
    assert a.key() != b.key()
    assert a.scope_hash() != b.scope_hash()


def test_make_key_still_works_and_matches_cache_request() -> None:
    """``make_key`` stays public; it must agree with the object it now delegates to."""
    msgs = _msgs()
    assert (
        make_key(msgs, "m", 0.5, provider="ollama", max_tokens=64)
        == CacheRequest(
            messages=msgs, model="m", temperature=0.5, provider="ollama", max_tokens=64
        ).key()
    )


def test_cache_request_content_cannot_forge_a_field_boundary() -> None:
    """Canonical JSON, not concatenation: content can never impersonate a message list."""
    two = CacheRequest(
        messages=[
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
        ],
        model="m",
        temperature=0.5,
    )
    one = CacheRequest(
        messages=[ChatMessage(role="user", content="a\x00user\x01b")],
        model="m",
        temperature=0.5,
    )
    assert two.key() != one.key()


# ---------------------------------------------------------------------------
# 1b. SqliteVecCache: honours the whole request, versions its schema
# ---------------------------------------------------------------------------

sqlite_vec = pytest.importorskip("sqlite_vec")

from actants.cache.semantic import (  # noqa: E402
    SCHEMA_VERSION,
    CacheSchemaMismatch,
    SqliteVecCache,
)


class StubEmbedder:
    """Deterministic embedder; identical text always gives an identical vector."""

    async def embed(self, text: str) -> list[float]:
        import math

        buckets = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            buckets[i % 8] += ord(ch)
        norm = math.sqrt(sum(b * b for b in buckets)) or 1.0
        return [b / norm for b in buckets]


def _cache(tmp_path, **kw) -> SqliteVecCache:
    return SqliteVecCache(
        tmp_path / "cache.db",
        StubEmbedder(),
        similarity_threshold=kw.pop("similarity_threshold", 0.5),
        default_ttl=kw.pop("default_ttl", None),
        **kw,
    )


@pytest.mark.asyncio
async def test_semantic_cache_does_not_share_entries_across_max_tokens(tmp_path) -> None:
    """The regression the review asked for, end to end through the real backend.

    Two requests identical except for ``max_tokens``. Before the fix the second one hit
    the first one's entry — a 4096-token answer served to a request capped at 16.
    """
    cache = _cache(tmp_path)
    msgs = _msgs("summarize the french revolution")

    short = CacheRequest(messages=msgs, model="m", temperature=0.7, max_tokens=16)
    long = CacheRequest(messages=msgs, model="m", temperature=0.7, max_tokens=4096)

    await cache.set_request(short, _result("terse"))
    assert (await cache.get_request(long)) is None, (
        "a request with a different max_tokens must not read another one's entry"
    )
    # ...and the original still hits, so we broke sharing, not caching.
    hit = await cache.get_request(short)
    assert hit is not None and hit.content == "terse"
    cache.close()


@pytest.mark.asyncio
async def test_semantic_cache_does_not_share_entries_across_providers(tmp_path) -> None:
    cache = _cache(tmp_path)
    msgs = _msgs("hello")
    await cache.set_request(
        CacheRequest(messages=msgs, model="m", temperature=0.7, provider="ollama"),
        _result("from ollama"),
    )
    miss = await cache.get_request(
        CacheRequest(messages=msgs, model="m", temperature=0.7, provider="openai")
    )
    assert miss is None
    cache.close()


@pytest.mark.asyncio
async def test_semantic_cache_does_not_share_entries_across_tools(tmp_path) -> None:
    cache = _cache(tmp_path)
    msgs = _msgs("hello")
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    await cache.set_request(
        CacheRequest(messages=msgs, model="m", temperature=0.7), _result("no tools")
    )
    miss = await cache.get_request(
        CacheRequest(messages=msgs, model="m", temperature=0.7, tools=[spec])
    )
    assert miss is None
    cache.close()


@pytest.mark.asyncio
async def test_semantic_cache_does_not_share_entries_across_response_format(tmp_path) -> None:
    cache = _cache(tmp_path)
    msgs = _msgs("hello")
    await cache.set_request(
        CacheRequest(messages=msgs, model="m", temperature=0.7), _result("prose")
    )
    miss = await cache.get_request(
        CacheRequest(
            messages=msgs,
            model="m",
            temperature=0.7,
            response_format={"type": "json_object"},
        )
    )
    assert miss is None
    cache.close()


@pytest.mark.asyncio
async def test_scope_is_pruned_before_the_knn_search_not_after(tmp_path) -> None:
    """Scoping must be a partition key, not a post-filter on a KNN result.

    ``MATCH ... AND k = 1`` picks the nearest vector *first*; a scope filter applied
    afterwards discards it and returns nothing, so a correct entry that exists in the
    right scope is reported as a miss whenever some other scope holds a nearer vector.
    Here both entries embed identically, so the wrong-scope one can win the KNN.
    """
    cache = _cache(tmp_path)
    msgs = _msgs("identical prompt text")
    mine = CacheRequest(messages=msgs, model="m", temperature=0.7, max_tokens=16)
    theirs = CacheRequest(messages=msgs, model="m", temperature=0.7, max_tokens=4096)

    await cache.set_request(mine, _result("mine"))
    await cache.set_request(theirs, _result("theirs"))

    got = await cache.get_request(mine)
    assert got is not None and got.content == "mine"
    other = await cache.get_request(theirs)
    assert other is not None and other.content == "theirs"
    cache.close()


@pytest.mark.asyncio
async def test_semantic_cache_still_matches_similar_prompts(tmp_path) -> None:
    """The point of a semantic cache must survive the tightened scope."""
    cache = _cache(tmp_path, similarity_threshold=0.5)
    req = CacheRequest(messages=_msgs("what is the capital of france"), model="m", temperature=0.7)
    await cache.set_request(req, _result("Paris"))
    near = CacheRequest(
        messages=_msgs("what is the capital of france?"), model="m", temperature=0.7
    )
    hit = await cache.get_request(near)
    assert hit is not None and hit.content == "Paris"
    cache.close()


@pytest.mark.asyncio
async def test_stale_schema_is_discarded_not_misread(tmp_path) -> None:
    """An old cache file must never serve entries keyed on fewer fields.

    Simulates a file written by the previous schema: the old ``entries`` layout, no
    ``scope_hash`` column, and a stale ``user_version``.
    """
    path = tmp_path / "cache.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            temperature REAL NOT NULL,
            fingerprint TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO entries (model, temperature, fingerprint, result_json, created_at) "
        "VALUES ('m', 0.7, 'user: hello', ?, 0)",
        (_result("stale answer").model_dump_json(),),
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    conn.commit()
    conn.close()

    cache = _cache(tmp_path)
    # Opening it must not explode, and must not serve the stale entry.
    got = await cache.get_request(CacheRequest(messages=_msgs(), model="m", temperature=0.7))
    assert got is None
    assert len(cache) == 0
    cache.close()

    # The file is now stamped with the current version and usable.
    reopened = _cache(tmp_path)
    await reopened.set_request(
        CacheRequest(messages=_msgs(), model="m", temperature=0.7), _result("fresh")
    )
    hit = await reopened.get_request(CacheRequest(messages=_msgs(), model="m", temperature=0.7))
    assert hit is not None and hit.content == "fresh"
    reopened.close()


@pytest.mark.asyncio
async def test_stale_schema_can_be_made_an_error(tmp_path) -> None:
    path = tmp_path / "cache.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    conn.commit()
    conn.close()

    cache = _cache(tmp_path, on_schema_mismatch="error")
    with pytest.raises(CacheSchemaMismatch) as exc:
        await cache.get_request(CacheRequest(messages=_msgs(), model="m", temperature=0.7))
    message = str(exc.value)
    assert str(SCHEMA_VERSION) in message
    assert "on_schema_mismatch='reset'" in message


def test_schema_version_is_tied_to_key_version() -> None:
    """Changing what the key covers must invalidate on-disk caches automatically."""
    assert SCHEMA_VERSION == KEY_VERSION


def test_on_schema_mismatch_rejects_unknown_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="on_schema_mismatch"):
        SqliteVecCache(tmp_path / "c.db", StubEmbedder(), on_schema_mismatch="explode")


@pytest.mark.asyncio
async def test_fresh_file_is_stamped_with_current_version(tmp_path) -> None:
    cache = _cache(tmp_path)
    await cache.set_request(CacheRequest(messages=_msgs(), model="m", temperature=0.7), _result())
    cache.close()
    conn = sqlite3.connect(tmp_path / "cache.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


@pytest.mark.asyncio
async def test_llm_routes_max_tokens_into_the_semantic_cache(tmp_path) -> None:
    """The client must build the request, not a subset of it.

    Two ``complete()`` calls differing only in ``max_tokens`` must both hit the provider.
    """

    class CountingProvider(BaseLLMProvider):
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
            self.calls += 1
            return _result(f"reply-{self.calls}")

        async def stream(self, *a: Any, **kw: Any) -> AsyncIterator[str]:
            yield ""

        async def health(self) -> bool:
            return True

    provider = CountingProvider()
    cache = _cache(tmp_path)
    llm = LLM(provider=provider, model="m", cache=cache, tracing=False)

    await llm.complete("capital of france", max_tokens=16)
    await llm.complete("capital of france", max_tokens=4096)
    assert provider.calls == 2, "requests with different max_tokens shared a cache entry"

    # Same max_tokens twice really is one call, so caching still works.
    await llm.complete("capital of france", max_tokens=16)
    assert provider.calls == 2
    cache.close()


@pytest.mark.asyncio
async def test_llm_prefers_request_protocol_but_still_supports_key_backends() -> None:
    """Exact-match backends keep working through the plain ``CacheBackend`` protocol."""
    provider = FakeLLMProvider([_result("a"), _result("b")])
    cache = InMemoryCache(default_ttl=None)
    llm = LLM(provider=provider, model="m", cache=cache, tracing=False)

    first = await llm.complete("hello", max_tokens=16)
    second = await llm.complete("hello", max_tokens=16)
    assert first.content == second.content
    assert len(cache) == 1

    await llm.complete("hello", max_tokens=4096)
    assert len(cache) == 2, "max_tokens must reach the exact-match key too"


# ---------------------------------------------------------------------------
# 3. supports_tool_calls is enforced
# ---------------------------------------------------------------------------


class NoToolsProvider(BaseLLMProvider):
    """A provider that honestly declares it cannot call tools."""

    name = "no-tools"
    supports_tool_calls = False
    supports_streaming_tools = False

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
        return _result("ignored your tools")

    async def stream(self, messages, model, **kw) -> AsyncIterator[str]:
        yield "ignored"

    async def health(self) -> bool:
        return True


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register_function("add", "Add two integers", add)
    return registry


@pytest.mark.asyncio
async def test_complete_rejects_tools_on_a_provider_that_cannot_use_them() -> None:
    llm = LLM(provider=NoToolsProvider(), model="m", tracing=False)
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    with pytest.raises(ToolCallsNotSupportedError) as exc:
        await llm.complete("2+2?", tools=[spec])

    message = str(exc.value)
    # Names the problem...
    assert "'no-tools'" in message
    assert "'add'" in message
    # ...and names the fix.
    assert "supports_tool_calls = True" in message
    assert "ollama" in message


@pytest.mark.asyncio
async def test_run_agent_rejects_tools_before_burning_a_call() -> None:
    provider = NoToolsProvider()
    llm = LLM(provider=provider, model="m", tracing=False)
    with pytest.raises(ToolCallsNotSupportedError):
        await llm.run_agent("2+2?", _registry())


@pytest.mark.asyncio
async def test_stream_events_rejects_tools_and_names_the_streaming_flag() -> None:
    class TextOnlyStreamer(NoToolsProvider):
        supports_tool_calls = True  # can call tools, but not while streaming
        supports_streaming_tools = False

    llm = LLM(provider=TextOnlyStreamer(), model="m", tracing=False)
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    with pytest.raises(ToolCallsNotSupportedError) as exc:
        async for _ in llm.stream_events("hi", tools=[spec]):
            pass
    message = str(exc.value)
    assert "supports_streaming_tools = True" in message
    assert "streaming tool calls" in message


@pytest.mark.asyncio
async def test_error_lists_the_tools_and_truncates_long_lists() -> None:
    llm = LLM(provider=NoToolsProvider(), model="m", tracing=False)
    specs = [
        ToolSpec(name=f"tool_{i}", description="d", parameters={"type": "object"}) for i in range(5)
    ]
    with pytest.raises(ToolCallsNotSupportedError) as exc:
        await llm.complete("hi", tools=specs)
    message = str(exc.value)
    assert "(5 total)" in message
    assert "5 tool(s)" in message


@pytest.mark.asyncio
async def test_no_tools_means_no_error() -> None:
    """A tool-less provider must stay perfectly usable for plain completions."""
    llm = LLM(provider=NoToolsProvider(), model="m", tracing=False)
    result = await llm.complete("hi")
    assert result.content == "ignored your tools"


@pytest.mark.asyncio
async def test_empty_tool_list_is_not_an_error() -> None:
    llm = LLM(provider=NoToolsProvider(), model="m", tracing=False)
    assert (await llm.complete("hi", tools=[])).content == "ignored your tools"


@pytest.mark.asyncio
async def test_capable_provider_is_unaffected() -> None:
    llm = LLM(provider=FakeLLMProvider([_result("ok")]), model="m", tracing=False)
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object"})
    assert (await llm.complete("hi", tools=[spec])).content == "ok"


# ---------------------------------------------------------------------------
# 4. Streaming goes through the LLM layer
# ---------------------------------------------------------------------------


class FlakyStreamProvider(BaseLLMProvider):
    """Fails the first ``fail_times`` stream attempts before any event is emitted."""

    name = "flaky"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, fail_times: int = 1) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self.models_seen: list[str] = []
        self.temps_seen: list[float] = []

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
        return _result("complete")

    async def stream(self, messages, model, **kw) -> AsyncIterator[str]:
        yield "x"

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kw
    ) -> AsyncIterator[StreamEvent]:
        self.attempts += 1
        self.models_seen.append(model)
        self.temps_seen.append(temperature)
        if self.attempts <= self.fail_times:
            raise ConnectionError("stream failed to open")
        yield TextDelta(text="recovered")
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_stream_events_retries_a_failure_before_the_first_event() -> None:
    """Streaming now gets the client's retry policy, exactly like ``complete``."""
    provider = FlakyStreamProvider(fail_times=1)
    llm = LLM(
        provider=provider,
        model="m",
        tracing=False,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
    )
    events = [e async for e in llm.stream_events("hi")]
    assert provider.attempts == 2
    assert any(isinstance(e, TextDelta) and e.text == "recovered" for e in events)


@pytest.mark.asyncio
async def test_stream_events_does_not_retry_after_emitting() -> None:
    """Restarting mid-stream would splice two completions into one response."""

    class FailsMidStream(FlakyStreamProvider):
        async def stream_events(self, messages, model, **kw) -> AsyncIterator[StreamEvent]:
            self.attempts += 1
            yield TextDelta(text="partial")
            raise ConnectionError("died mid-stream")

    provider = FailsMidStream()
    llm = LLM(
        provider=provider,
        model="m",
        tracing=False,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
    )
    seen = []
    with pytest.raises(ConnectionError):
        async for event in llm.stream_events("hi"):
            seen.append(event)
    assert provider.attempts == 1, "retried after the consumer already saw output"
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_stream_events_gives_up_after_max_attempts() -> None:
    provider = FlakyStreamProvider(fail_times=99)
    llm = LLM(
        provider=provider,
        model="m",
        tracing=False,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, jitter=0.0),
    )
    with pytest.raises(ConnectionError):
        async for _ in llm.stream_events("hi"):
            pass
    assert provider.attempts == 2


@pytest.mark.asyncio
async def test_stream_events_honours_per_call_model_and_temperature() -> None:
    provider = FlakyStreamProvider(fail_times=0)
    llm = LLM(provider=provider, model="default-model", tracing=False)
    async for _ in llm.stream_events("hi", model="override-model", temperature=0.123):
        pass
    assert provider.models_seen == ["override-model"]
    assert provider.temps_seen == [0.123]


@pytest.mark.asyncio
async def test_stream_text_also_goes_through_the_layer() -> None:
    provider = FlakyStreamProvider(fail_times=1)
    llm = LLM(
        provider=provider,
        model="m",
        tracing=False,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
    )
    chunks = [c async for c in llm.stream("hi")]
    assert "".join(chunks) == "recovered"
    assert provider.attempts == 2


# ---------------------------------------------------------------------------
# 2. Agent concurrency
# ---------------------------------------------------------------------------


class SlowProvider(BaseLLMProvider):
    """Yields to the event loop mid-call so concurrent runs genuinely interleave.

    Echoes back which prompt it saw, so a test can prove a run's history was not
    contaminated by another run's messages.
    """

    name = "slow"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.seen: list[list[ChatMessage]] = []

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
        self.seen.append(list(messages))
        await asyncio.sleep(self.delay)
        user = [m.content for m in messages if m.role == "user"]
        return _result(f"answer-to-{user[-1]}")

    async def stream(self, messages, model, **kw) -> AsyncIterator[str]:
        result = await self.complete(messages, model)
        yield result.content

    async def stream_events(self, messages, model, **kw) -> AsyncIterator[StreamEvent]:
        result = await self.complete(messages, model)
        for ch in result.content:
            yield TextDelta(text=ch)
            await asyncio.sleep(0)
        yield UsageDelta(usage=result.usage, cost_usd=0.0)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_interleave_history() -> None:
    """Three genuinely concurrent runs; none may see another's user message.

    Before the fix all three appended to one ConversationMemory, so each step sent the
    merged history and the answers were computed from a conversation no caller asked for.
    """
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))

    results = await asyncio.gather(
        agent.run("alpha"),
        agent.run("beta"),
        agent.run("gamma"),
    )

    assert [r.content for r in results] == [
        "answer-to-alpha",
        "answer-to-beta",
        "answer-to-gamma",
    ]

    # Each request the provider saw must contain exactly one user message.
    for messages in provider.seen:
        users = [m.content for m in messages if m.role == "user"]
        assert len(users) == 1, f"run saw a contaminated history: {users}"


@pytest.mark.asyncio
async def test_concurrent_runs_all_commit_their_turn() -> None:
    """Isolation must not lose turns: every run's messages land in the agent's memory."""
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))

    await asyncio.gather(agent.run("alpha"), agent.run("beta"), agent.run("gamma"))

    contents = [m.content for m in agent.memory.messages()]
    assert sorted(c for c in contents if not c.startswith("answer-")) == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert len([c for c in contents if c.startswith("answer-to-")]) == 3


@pytest.mark.asyncio
async def test_each_turn_is_committed_contiguously() -> None:
    """No run may observe, or leave behind, a half-written turn."""
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))
    await asyncio.gather(agent.run("alpha"), agent.run("beta"))

    messages = agent.memory.messages()
    # Every user message is immediately followed by its own answer.
    for i, m in enumerate(messages):
        if m.role == "user":
            assert messages[i + 1].role == "assistant"
            assert messages[i + 1].content == f"answer-to-{m.content}"


@pytest.mark.asyncio
async def test_sequential_runs_still_remember_context() -> None:
    """Isolation is per-run, not per-call: turn 2 must still see turn 1."""
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))

    await agent.run("first")
    await agent.run("second")

    last_request = provider.seen[-1]
    contents = [m.content for m in last_request]
    assert "first" in contents
    assert "answer-to-first" in contents
    assert "second" in contents


@pytest.mark.asyncio
async def test_failed_run_commits_nothing() -> None:
    """A run that raises must not strand its user message in the conversation."""

    class BoomProvider(SlowProvider):
        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):
            raise RuntimeError("provider exploded")

    agent = Agent(llm=LLM(provider=BoomProvider(), model="m", tracing=False))
    with pytest.raises(RuntimeError, match="provider exploded"):
        await agent.run("doomed")
    assert agent.memory.messages() == []


@pytest.mark.asyncio
async def test_serialized_mode_makes_each_run_see_the_previous_one() -> None:
    """The other contract must be reachable deliberately, and must actually serialize."""
    provider = SlowProvider()
    agent = Agent(
        llm=LLM(provider=provider, model="m", tracing=False),
        concurrency="serialized",
    )

    await asyncio.gather(agent.run("alpha"), agent.run("beta"), agent.run("gamma"))

    # Runs queued, so the last one saw all three user messages.
    user_counts = sorted(len([m for m in seen if m.role == "user"]) for seen in provider.seen)
    assert user_counts == [1, 2, 3]


@pytest.mark.asyncio
async def test_serialized_mode_preserves_total_order() -> None:
    provider = SlowProvider()
    agent = Agent(
        llm=LLM(provider=provider, model="m", tracing=False),
        concurrency="serialized",
    )
    await asyncio.gather(agent.run("alpha"), agent.run("beta"))
    roles = [m.role for m in agent.memory.messages()]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_unknown_concurrency_mode_is_rejected_with_the_alternatives() -> None:
    with pytest.raises(ValueError) as exc:
        Agent(llm=LLM(provider=FakeLLMProvider(), model="m"), concurrency="yolo")
    message = str(exc.value)
    assert "isolated" in message and "serialized" in message


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_interleave_history() -> None:
    """stream() carries the same guarantee as run()."""
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))

    async def drain(prompt: str) -> str:
        parts = []
        async for event in agent.stream(prompt):
            if hasattr(event, "text"):
                parts.append(event.text)
        return "".join(parts)

    out = await asyncio.gather(drain("alpha"), drain("beta"), drain("gamma"))
    assert sorted(out) == ["answer-to-alpha", "answer-to-beta", "answer-to-gamma"]
    for messages in provider.seen:
        assert len([m for m in messages if m.role == "user"]) == 1


@pytest.mark.asyncio
async def test_abandoned_stream_commits_nothing() -> None:
    """Consumer stops iterating part-way: the agent's memory must be untouched."""
    provider = SlowProvider()
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))

    stream = agent.stream("partial")
    await stream.__anext__()
    await stream.aclose()

    assert agent.memory.messages() == []


@pytest.mark.asyncio
async def test_agent_memory_is_the_seed_for_isolated_runs() -> None:
    """A run must inherit whatever was in memory when it started."""
    provider = SlowProvider()
    memory = ConversationMemory(system="be terse")
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False), memory=memory)

    await agent.run("hello")
    assert provider.seen[0][0].role == "system"
    assert provider.seen[0][0].content == "be terse"


@pytest.mark.asyncio
async def test_agent_stream_uses_the_llm_layer_not_the_provider() -> None:
    """The defect: ``Agent.stream`` called ``provider.stream_events`` directly.

    Proven by making the LLM layer observable — a run whose stream never touches the
    layer would not increment this counter.
    """
    provider = FakeLLMProvider([fake_completion("hello")])
    llm = LLM(provider=provider, model="m", tracing=False)

    calls = 0
    original = llm.stream_events

    def counting(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    llm.stream_events = counting  # type: ignore[method-assign]
    agent = Agent(llm=llm)
    async for _ in agent.stream("hi"):
        pass
    assert calls == 1


@pytest.mark.asyncio
async def test_agent_stream_inherits_retry_from_the_llm() -> None:
    """Before the fix a streamed run had no retry at all, while run() did."""
    provider = FlakyStreamProvider(fail_times=1)
    agent = Agent(
        llm=LLM(
            provider=provider,
            model="m",
            tracing=False,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
        )
    )
    texts = [e.text async for e in agent.stream("hi") if isinstance(e, TextDelta | AgentTextDelta)]
    assert provider.attempts == 2
    assert "".join(texts) == "recovered"


@pytest.mark.asyncio
async def test_agent_stream_and_run_agree_on_final_content() -> None:
    """A streamed run and a non-streamed run must not behave differently."""
    streamed = Agent(
        llm=LLM(
            provider=FakeLLMProvider([fake_completion("same answer")]), model="m", tracing=False
        )
    )
    plain = Agent(
        llm=LLM(
            provider=FakeLLMProvider([fake_completion("same answer")]), model="m", tracing=False
        )
    )

    events = [e async for e in streamed.stream("hi")]
    completed = [e for e in events if isinstance(e, AgentRunCompleted)]
    result = await plain.run("hi")

    assert completed[0].content == result.content == "same answer"
    assert [m.content for m in streamed.memory.messages()] == [
        m.content for m in plain.memory.messages()
    ]


@pytest.mark.asyncio
async def test_agent_stream_still_dispatches_tools() -> None:
    """Routing through the LLM layer must not break the tool loop."""
    provider = FakeLLMProvider(
        [
            fake_tool_call_completion("add", {"a": 2, "b": 3}, call_id="t1"),
            fake_completion("Result is 5"),
        ]
    )
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False), tools=_registry())
    events = [e async for e in agent.stream("2 + 3?")]

    started = [e for e in events if isinstance(e, AgentToolCallStarted)]
    completed = [e for e in events if isinstance(e, AgentToolCallCompleted)]
    assert len(started) == 1 and started[0].call.name == "add"
    assert len(completed) == 1 and completed[0].ok
    assert completed[0].value == 5
