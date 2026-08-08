"""Checkpoint / resume / human-in-the-loop.

The guarantee under test: resume is at-most-once for every tool call whose result was
recorded, and at-least-once for the single call that was in flight when the run died.
Most tests here assert against a call counter, because "did not re-run" is the whole
product and a message-shape assertion would not catch a duplicated side effect.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from actants.agents.agent import Agent, AgentResult
from actants.agents.checkpoint import (
    SCHEMA_VERSION,
    Checkpoint,
    Checkpointer,
    InMemoryCheckpointer,
    SqliteCheckpointer,
)
from actants.errors import (
    ActantsError,
    CheckpointSchemaMismatch,
    UnknownThreadError,
    UnresolvedToolCallError,
)
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from actants.llm.client import LLM
from actants.tools.registry import ToolRegistry


def _completion(content: str, tool_calls: list[ToolCall] | None = None) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="test",
        provider="scripted",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        tool_calls=tool_calls or [],
    )


def _call(name: str, call_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


class ScriptedProvider(BaseLLMProvider):
    """Pops one scripted completion per call, recording the messages it saw.

    A scripted entry may be an exception instance, which is raised instead — that is how
    these tests simulate a process dying at a chosen point in the loop.
    """

    name = "scripted"
    supports_tool_calls = True

    def __init__(self, responses: list[CompletionResult | Exception]) -> None:
        self._responses: list[CompletionResult | Exception] = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def queue(self, *responses: CompletionResult | Exception) -> None:
        self._responses.extend(responses)

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
        self.calls.append(list(messages))
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def stream_events(self, messages, model, **kwargs) -> AsyncIterator[Any]:  # type: ignore[no-untyped-def]
        raise NotImplementedError
        yield  # pragma: no cover

    async def health(self) -> bool:
        return True


class Counter:
    """Hands out a tool handler that records every dispatch.

    ``count`` is the assertion that matters throughout this file: a duplicated side
    effect shows up here and nowhere else. The handler is a real ``async def`` rather
    than ``__call__``, because ``register_function`` rejects anything
    ``inspect.iscoroutinefunction`` does not recognize.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def handler(self) -> Callable[..., Awaitable[str]]:
        async def run(**kwargs: Any) -> str:
            self.calls.append(dict(kwargs))
            return f"ran-{len(self.calls)}"

        return run

    @property
    def count(self) -> int:
        return len(self.calls)


def _agent(
    provider: ScriptedProvider,
    tools: ToolRegistry | None = None,
    **kwargs: Any,
) -> Agent:
    return Agent(llm=LLM(provider=provider, model="m", tracing=False), tools=tools, **kwargs)


def _slow(counter: Counter) -> Callable[..., Awaitable[str]]:
    """A handler that yields to the loop mid-dispatch, as any real I/O tool does.

    The concurrency regressions need that suspension point: without one, ``gather`` runs
    the two resumes end to end and the race they exist to catch never opens.
    """

    async def run(**kwargs: Any) -> str:
        counter.calls.append(dict(kwargs))
        await asyncio.sleep(0.02)
        return "sent"

    return run


def _registry(**counters: Counter) -> ToolRegistry:
    """Build a registry from named counters; an ``unsafe_`` prefix means non-idempotent."""
    registry = ToolRegistry()
    for name, counter in counters.items():
        registry.register_function(
            name,
            f"tool {name}",
            counter.handler,
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            idempotent=not name.startswith("unsafe_"),
        )
    return registry


# ---------------------------------------------------------------------------
# Backward compatibility: nothing changes when durability is not asked for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_without_a_checkpointer_is_unchanged() -> None:
    provider = ScriptedProvider(
        [
            _completion("", [_call("lookup", "c1", x=1)]),
            _completion("done"),
        ]
    )
    lookup = Counter()
    agent = _agent(provider, _registry(lookup=lookup))

    result = await agent.run("go")

    assert result.content == "done"
    assert result.interrupted is False
    assert result.pending_call is None
    assert result.thread_id is None
    assert lookup.count == 1
    assert [m.role for m in agent.memory.messages()] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_checkpointer_without_a_thread_id_persists_nothing() -> None:
    """Durability is opt-in per run, not a mode the agent is in."""
    provider = ScriptedProvider([_completion("done")])
    store = InMemoryCheckpointer()
    agent = _agent(provider, checkpointer=store)

    await agent.run("go")

    assert await store.list_threads() == []


@pytest.mark.asyncio
async def test_thread_id_without_a_checkpointer_is_rejected() -> None:
    agent = _agent(ScriptedProvider([_completion("done")]))
    with pytest.raises(ValueError, match="checkpointer"):
        await agent.run("go", thread_id="t1")


@pytest.mark.asyncio
async def test_concurrent_durable_runs_keep_isolated_histories() -> None:
    """Durability must not regress the concurrency contract."""

    class Yielding(ScriptedProvider):
        async def complete(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
            # Suspends mid-call so the two runs genuinely overlap; without this they
            # simply execute one after the other and prove nothing.
            await asyncio.sleep(0.01)
            return await super().complete(messages, model, **kwargs)

    provider = Yielding([_completion("a"), _completion("b")])
    store = InMemoryCheckpointer()
    agent = _agent(provider, checkpointer=store)

    await asyncio.gather(agent.run("alpha", thread_id="t1"), agent.run("beta", thread_id="t2"))

    for messages in provider.calls:
        assert len([m for m in messages if m.role == "user"]) == 1
    assert sorted(await store.list_threads()) == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Checkpoint contents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_run_is_checkpointed_as_completed() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), _completion("done")]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)

    await agent.run("go", thread_id="t1")

    checkpoint = await store.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "completed"
    assert checkpoint.thread_id == "t1"
    assert [m.role for m in checkpoint.messages] == ["user", "assistant", "tool", "assistant"]
    assert len(checkpoint.steps) == 2
    assert checkpoint.steps[0].tool_results == ['"ran-1"']


@pytest.mark.asyncio
async def test_each_tool_result_is_checkpointed_individually() -> None:
    """The boundary the guarantee rests on: a write per result, not per step."""
    provider = ScriptedProvider(
        [
            _completion(
                "",
                [
                    _call("lookup", "c1", x=1),
                    _call("lookup", "c2", x=2),
                    _call("lookup", "c3", x=3),
                ],
            ),
            _completion("done"),
        ]
    )
    seen: list[tuple[str, int]] = []

    class Recording(InMemoryCheckpointer):
        async def put(self, checkpoint: Checkpoint) -> None:
            done = sum(len(s.tool_results) for s in checkpoint.steps)
            seen.append((checkpoint.status, done))
            await super().put(checkpoint)

    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=Recording())
    await agent.run("go", thread_id="t1")

    # One write before any tool ran, then one after each of the three results.
    assert seen[:4] == [("running", 0), ("running", 1), ("running", 2), ("running", 3)]
    assert seen[-1][0] == "completed"


@pytest.mark.asyncio
async def test_failed_run_is_checkpointed_as_failed_and_re_raises() -> None:
    provider = ScriptedProvider([RuntimeError("provider exploded")])
    store = InMemoryCheckpointer()
    agent = _agent(provider, checkpointer=store)

    with pytest.raises(RuntimeError, match="provider exploded"):
        await agent.run("go", thread_id="t1")

    # Nothing was checkpointed before the first completion, so there is nothing to mark.
    assert await store.get("t1") is None


@pytest.mark.asyncio
async def test_failure_after_a_tool_ran_marks_the_thread_failed() -> None:
    provider = ScriptedProvider(
        [
            _completion("", [_call("lookup", "c1", x=1)]),
            RuntimeError("provider exploded"),
        ]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)

    with pytest.raises(RuntimeError, match="provider exploded"):
        await agent.run("go", thread_id="t1")

    checkpoint = await store.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "failed"
    assert "provider exploded" in (checkpoint.error or "")


# ---------------------------------------------------------------------------
# Resume after a crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_after_a_crash_continues_the_run() -> None:
    """The headline case: the process dies mid-run, a fresh call picks it up."""
    provider = ScriptedProvider(
        [
            _completion("", [_call("lookup", "c1", x=1)]),
            RuntimeError("process died"),
        ]
    )
    store = InMemoryCheckpointer()
    lookup = Counter()
    agent = _agent(provider, _registry(lookup=lookup), checkpointer=store)

    with pytest.raises(RuntimeError, match="process died"):
        await agent.run("go", thread_id="t1")

    # The store is all that survives; a fresh Agent object stands in for a new process.
    stored = await store.get("t1")
    assert stored is not None
    stored.status = "running"
    stored.error = None
    await store.put(stored)

    provider.queue(_completion("finished"))
    resumed = _agent(provider, _registry(lookup=lookup), checkpointer=store)
    result = await resumed.resume("t1")

    assert result.content == "finished"
    assert result.thread_id == "t1"
    assert lookup.count == 1, "the tool that already ran was dispatched again"


@pytest.mark.asyncio
async def test_resume_does_not_rerun_a_completed_tool_call() -> None:
    """A step with three calls dies after the second; only the third is dispatched.

    The crash is staged in the checkpointer rather than the tool, because
    ``ToolRegistry.call`` turns a handler exception into a failed ``ToolResult`` instead
    of letting it escape — which is a recorded result, not a dead process.
    """

    class DieAfterTwo(InMemoryCheckpointer):
        async def put(self, checkpoint: Checkpoint) -> None:
            await super().put(checkpoint)
            if sum(len(s.tool_results) for s in checkpoint.steps) == 2:
                raise RuntimeError("process died")

    calls = [_call("lookup", "c1", x=1), _call("lookup", "c2", x=2), _call("lookup", "c3", x=3)]
    provider = ScriptedProvider([_completion("", calls)])
    lookup = Counter()
    dying = DieAfterTwo()
    agent = _agent(provider, _registry(lookup=lookup), checkpointer=dying)

    with pytest.raises(RuntimeError, match="process died"):
        await agent.run("go", thread_id="t1")
    assert lookup.count == 2

    stored = await dying.get("t1")
    assert stored is not None
    assert sum(len(s.tool_results) for s in stored.steps) == 2
    stored.status = "running"
    stored.error = None
    store = InMemoryCheckpointer()
    await store.put(stored)

    provider.queue(_completion("finished"))
    resumed = _agent(provider, _registry(lookup=lookup), checkpointer=store)
    result = await resumed.resume("t1")

    assert result.content == "finished"
    assert lookup.count == 3, "the third call must run exactly once"
    assert [c["x"] for c in lookup.calls] == [1, 2, 3]


@pytest.mark.asyncio
async def test_resume_replays_recorded_results_into_the_next_prompt() -> None:
    """The resumed LLM call must see the tool results the crashed run recorded."""
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), RuntimeError("died")]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)
    with pytest.raises(RuntimeError):
        await agent.run("go", thread_id="t1")

    stored = await store.get("t1")
    assert stored is not None
    stored.status = "running"
    await store.put(stored)

    provider.queue(_completion("finished"))
    await _agent(provider, _registry(lookup=Counter()), checkpointer=store).resume("t1")

    last = provider.calls[-1]
    assert [m.role for m in last] == ["user", "assistant", "tool"]
    assert last[-1].content == '"ran-1"'
    assert last[-1].tool_call_id == "c1"


@pytest.mark.asyncio
async def test_resume_commits_the_whole_turn_to_agent_memory_once() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), RuntimeError("died")]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)
    with pytest.raises(RuntimeError):
        await agent.run("go", thread_id="t1")
    assert agent.memory.messages() == [], "a crashed turn must commit nothing"

    stored = await store.get("t1")
    assert stored is not None
    stored.status = "running"
    await store.put(stored)

    provider.queue(_completion("finished"))
    await agent.resume("t1")

    assert [m.role for m in agent.memory.messages()] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_resume_honours_the_original_step_budget() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), RuntimeError("died")]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)
    with pytest.raises(RuntimeError):
        await agent.run("go", thread_id="t1", max_steps=2)

    stored = await store.get("t1")
    assert stored is not None
    assert stored.max_steps == 2
    stored.status = "running"
    await store.put(stored)

    # Step 0 is spent; the budget of 2 leaves exactly one more, which asks for a tool
    # again and so exhausts the run.
    provider.queue(_completion("", [_call("lookup", "c2", x=2)]))
    with pytest.raises(RuntimeError, match="max_steps=2"):
        await agent.resume("t1")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def _stage_in_flight(tool: str) -> tuple[InMemoryCheckpointer, ScriptedProvider, Counter]:
    """Leave thread ``t1`` checkpointed with call ``c1`` in flight and no result for it.

    The surviving state is moved into a clean store, which is what a new process sees:
    the crash mechanism belongs to the process that died, not to the file it left behind.
    """
    provider = ScriptedProvider([_completion("", [_call(tool, "c1", x=1)])])
    counter = Counter()

    class DieBeforeDispatch(InMemoryCheckpointer):
        async def put(self, checkpoint: Checkpoint) -> None:
            await super().put(checkpoint)
            if not checkpoint.steps[-1].tool_results:
                raise RuntimeError("process died")

    dying = DieBeforeDispatch()
    agent = _agent(provider, _registry(**{tool: counter}), checkpointer=dying)
    with pytest.raises(RuntimeError, match="process died"):
        await agent.run("go", thread_id="t1")
    assert counter.count == 0

    stored = await dying.get("t1")
    assert stored is not None
    stored.status = "running"
    stored.error = None
    store = InMemoryCheckpointer()
    await store.put(stored)
    return store, provider, counter


@pytest.mark.asyncio
async def test_in_flight_non_idempotent_tool_is_not_auto_replayed() -> None:
    store, provider, send = await _stage_in_flight("unsafe_send")

    with pytest.raises(UnresolvedToolCallError) as exc:
        await _agent(provider, _registry(unsafe_send=send), checkpointer=store).resume("t1")

    assert send.count == 0, "a non-idempotent in-flight call must not be re-dispatched"
    assert exc.value.call.id == "c1"
    assert exc.value.call.name == "unsafe_send"
    assert exc.value.thread_id == "t1"
    assert "idempotent=False" in str(exc.value)
    assert "resolve='retry'" in str(exc.value)


@pytest.mark.asyncio
async def test_in_flight_idempotent_tool_is_replayed() -> None:
    """The other half of the contract: the default really does re-dispatch."""
    store, provider, lookup = await _stage_in_flight("lookup")

    provider.queue(_completion("finished"))
    result = await _agent(provider, _registry(lookup=lookup), checkpointer=store).resume("t1")

    assert result.content == "finished"
    assert lookup.count == 1


@pytest.mark.asyncio
async def test_unresolved_call_can_be_retried_explicitly() -> None:
    store, provider, send = await _stage_in_flight("unsafe_send")
    provider.queue(_completion("finished"))
    result = await _agent(provider, _registry(unsafe_send=send), checkpointer=store).resume(
        "t1", resolve="retry"
    )
    assert result.content == "finished"
    assert send.count == 1


@pytest.mark.asyncio
async def test_unresolved_call_can_be_skipped_explicitly() -> None:
    store, provider, send = await _stage_in_flight("unsafe_send")
    provider.queue(_completion("finished"))
    result = await _agent(provider, _registry(unsafe_send=send), checkpointer=store).resume(
        "t1", resolve="skip"
    )
    assert result.content == "finished"
    assert send.count == 0
    tool_msg = [m for m in result.messages if m.role == "tool"][-1]
    assert "not safe to repeat" in tool_msg.content
    assert tool_msg.tool_call_id == "c1"
    # The model got the skip as an ordinary tool result and answered it.
    assert provider.calls[-1][-1].role == "tool"


@pytest.mark.asyncio
async def test_in_flight_call_to_a_tool_the_registry_lost_is_not_replayed() -> None:
    """Regression: a renamed tool used to discard the ``idempotent=False`` safety net.

    ``self.tools.get`` raises for a tool that is gone, and the suppressed ToolError left
    ``idempotent`` at its True default — so instead of raising, the run finished and told
    the model "Unknown tool", i.e. that the side effect had NOT happened, when it may
    well have. An unknown tool is exactly the case actants cannot vouch for.
    """
    store, provider, send = await _stage_in_flight("unsafe_send")

    renamed = Counter()
    registry = _registry(unsafe_send_v2=renamed)
    provider.queue(_completion("finished"))

    with pytest.raises(UnresolvedToolCallError) as exc:
        await _agent(provider, registry, checkpointer=store).resume("t1")

    assert send.count == 0 and renamed.count == 0
    assert exc.value.call.name == "unsafe_send"
    assert "no longer has" in str(exc.value)


@pytest.mark.asyncio
async def test_concurrent_resumes_of_one_thread_dispatch_the_call_once() -> None:
    """Regression: two resumes in one process both dispatched the in-flight call.

    The checkpointer's lock only serializes individual ``put`` calls, not the
    read-decide-dispatch sequence, so both resumes read the same "running" checkpoint and
    each dispatched. Asserted on the dispatch count — the consequence — because both
    resumes returning an AgentResult looked perfectly healthy.
    """
    store, provider, send = await _stage_in_flight("unsafe_send")
    provider.queue(_completion("finished"), _completion("finished"))
    registry = ToolRegistry()
    registry.register_function("unsafe_send", "send", _slow(send), idempotent=False)
    agent = _agent(provider, registry, checkpointer=store)

    results = await asyncio.gather(
        agent.resume("t1", resolve="retry"),
        agent.resume("t1", resolve="retry"),
        return_exceptions=True,
    )

    assert send.count == 1, "the in-flight non-idempotent call was dispatched twice"
    assert any(isinstance(r, AgentResult) for r in results)


@pytest.mark.asyncio
async def test_concurrent_approvals_of_one_interrupt_dispatch_the_call_once() -> None:
    """The same race on the interrupt path: approving twice must charge once."""
    provider = ScriptedProvider([_completion("", [_call("unsafe_send", "c1", x=1)])])
    charge = Counter()
    store = InMemoryCheckpointer()
    registry = ToolRegistry()
    registry.register_function("unsafe_send", "send", _slow(charge), idempotent=False)
    agent = _agent(
        provider,
        registry,
        checkpointer=store,
        interrupt_before=["unsafe_send"],
    )
    paused = await agent.run("go", thread_id="t1")
    assert paused.interrupted and charge.count == 0

    provider.queue(_completion("finished"), _completion("finished"))
    await asyncio.gather(
        agent.resume("t1", approve=True),
        agent.resume("t1", approve=True),
        return_exceptions=True,
    )

    assert charge.count == 1, "approving concurrently dispatched the guarded tool twice"


@pytest.mark.asyncio
async def test_resumes_of_different_threads_are_not_serialized_against_each_other() -> None:
    """The resume lock is per thread_id, so unrelated threads still overlap."""
    provider = ScriptedProvider([])
    agent = _agent(provider, checkpointer=InMemoryCheckpointer())

    first = agent._resume_guard("a")
    assert agent._resume_guard("a") is first, "one lock per thread_id"
    assert agent._resume_guard("b") is not first


@pytest.mark.asyncio
async def test_resume_rejects_an_unknown_resolution() -> None:
    store, provider, send = await _stage_in_flight("unsafe_send")
    agent = _agent(provider, _registry(unsafe_send=send), checkpointer=store)
    with pytest.raises(ValueError, match="resolve must be one of"):
        await agent.resume("t1", resolve="yolo")  # type: ignore[arg-type]


def test_tools_are_idempotent_by_default() -> None:
    registry = ToolRegistry()

    async def read(x: int) -> int:
        return x

    tool = registry.register_function("read", "read", read)
    assert tool.idempotent is True

    async def write(x: int) -> int:
        return x

    unsafe = registry.register_function("write", "write", write, idempotent=False)
    assert unsafe.idempotent is False


# ---------------------------------------------------------------------------
# Human in the loop
# ---------------------------------------------------------------------------


async def _interrupted_run() -> tuple[Agent, ScriptedProvider, Counter, AgentResult]:
    provider = ScriptedProvider([_completion("about to send", [_call("unsafe_send", "c1", x=1)])])
    send = Counter()
    agent = _agent(
        provider,
        _registry(unsafe_send=send),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )
    result = await agent.run("go", thread_id="t1")
    return agent, provider, send, result


@pytest.mark.asyncio
async def test_interrupt_pauses_before_dispatch() -> None:
    agent, _provider, send, result = await _interrupted_run()

    assert result.interrupted is True
    assert result.pending_call is not None
    assert result.pending_call.name == "unsafe_send"
    assert result.pending_call.id == "c1"
    assert result.thread_id == "t1"
    assert result.content == "about to send"
    assert send.count == 0

    assert agent.checkpointer is not None
    checkpoint = await agent.checkpointer.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "interrupted"
    assert checkpoint.pending_call is not None
    assert checkpoint.pending_call.id == "c1"


@pytest.mark.asyncio
async def test_interrupted_turn_is_not_committed_to_memory() -> None:
    """A pause leaves a half-written turn; the agent's memory must not see it."""
    agent, _provider, _send, _result = await _interrupted_run()
    assert agent.memory.messages() == []


@pytest.mark.asyncio
async def test_interrupt_then_approve_dispatches_and_continues() -> None:
    agent, provider, send, _result = await _interrupted_run()
    provider.queue(_completion("sent it"))

    resumed = await agent.resume("t1", approve=True)

    assert resumed.content == "sent it"
    assert resumed.interrupted is False
    assert send.count == 1
    assert send.calls == [{"x": 1}]
    assert [m.role for m in agent.memory.messages()] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_interrupt_then_reject_lets_the_model_respond() -> None:
    agent, provider, send, _result = await _interrupted_run()
    provider.queue(_completion("understood, I won't send it"))

    resumed = await agent.resume("t1", approve=False)

    assert resumed.content == "understood, I won't send it"
    assert send.count == 0, "a rejected call must not run"
    rejected = [m for m in resumed.messages if m.role == "tool"]
    assert len(rejected) == 1
    assert "rejected by a human" in rejected[0].content
    assert rejected[0].tool_call_id == "c1"
    # The model saw the rejection and got to answer it, rather than the run dying.
    assert provider.calls[-1][-1].role == "tool"


@pytest.mark.asyncio
async def test_resuming_an_interrupted_thread_without_a_decision_is_an_error() -> None:
    agent, _provider, _send, _result = await _interrupted_run()
    with pytest.raises(ValueError) as exc:
        await agent.resume("t1")
    assert "approve=True" in str(exc.value)
    assert "unsafe_send" in str(exc.value)


@pytest.mark.asyncio
async def test_interrupt_pauses_mid_step_and_keeps_earlier_results() -> None:
    """A guarded tool in the middle of a step: the calls before it must not be redone."""
    provider = ScriptedProvider(
        [
            _completion(
                "",
                [
                    _call("lookup", "c1", x=1),
                    _call("unsafe_send", "c2", x=2),
                    _call("lookup", "c3", x=3),
                ],
            )
        ]
    )
    lookup, send = Counter(), Counter()
    agent = _agent(
        provider,
        _registry(lookup=lookup, unsafe_send=send),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )

    paused = await agent.run("go", thread_id="t1")
    assert paused.interrupted
    assert paused.pending_call is not None and paused.pending_call.id == "c2"
    assert lookup.count == 1, "only the call before the guarded one may have run"
    assert send.count == 0

    provider.queue(_completion("all done"))
    result = await agent.resume("t1", approve=True)

    assert result.content == "all done"
    assert send.count == 1
    assert lookup.count == 2, "c1 must not be re-dispatched, and c3 must run"
    assert [c["x"] for c in lookup.calls] == [1, 3]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_two_guarded_calls_in_one_step_pause_twice() -> None:
    """An approved pause must not make the *next* guarded call look like a crash.

    The regression: approving c1 fell straight into the in-flight-crash path for c2 and
    raised UnresolvedToolCallError, even though c2 had provably never started.
    """
    provider = ScriptedProvider(
        [_completion("", [_call("unsafe_send", "c1", x=1), _call("unsafe_send", "c2", x=2)])]
    )
    send = Counter()
    agent = _agent(
        provider,
        _registry(unsafe_send=send),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )

    first = await agent.run("go", thread_id="t1")
    assert first.pending_call is not None and first.pending_call.id == "c1"

    second = await agent.resume("t1", approve=True)
    assert second.interrupted is True
    assert second.pending_call is not None and second.pending_call.id == "c2"
    assert send.count == 1

    provider.queue(_completion("both sent"))
    result = await agent.resume("t1", approve=True)
    assert result.content == "both sent"
    assert [c["x"] for c in send.calls] == [1, 2]


@pytest.mark.asyncio
async def test_interrupt_reject_still_runs_the_remaining_calls() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("unsafe_send", "c1", x=1), _call("lookup", "c2", x=2)])]
    )
    lookup, send = Counter(), Counter()
    agent = _agent(
        provider,
        _registry(lookup=lookup, unsafe_send=send),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )
    await agent.run("go", thread_id="t1")

    provider.queue(_completion("noted"))
    result = await agent.resume("t1", approve=False)

    assert result.content == "noted"
    assert send.count == 0
    assert lookup.count == 1, "rejecting one call must not abandon the rest of the step"


@pytest.mark.asyncio
async def test_interrupt_only_fires_for_named_tools() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), _completion("done")]
    )
    lookup = Counter()
    agent = _agent(
        provider,
        _registry(lookup=lookup, unsafe_send=Counter()),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )
    result = await agent.run("go", thread_id="t1")
    assert result.interrupted is False
    assert lookup.count == 1


@pytest.mark.asyncio
async def test_interrupt_without_a_thread_id_is_an_error() -> None:
    """There would be nothing to resume from, so refuse rather than silently dispatch."""
    provider = ScriptedProvider([_completion("", [_call("unsafe_send", "c1", x=1)])])
    send = Counter()
    agent = _agent(
        provider,
        _registry(unsafe_send=send),
        checkpointer=InMemoryCheckpointer(),
        interrupt_before=["unsafe_send"],
    )
    with pytest.raises(ValueError, match="interrupt_before"):
        await agent.run("go")
    assert send.count == 0


def test_interrupt_before_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError, match="one character at a time"):
        Agent(
            llm=LLM(provider=ScriptedProvider([]), model="m", tracing=False),
            interrupt_before="send_email",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Terminal and unknown threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resuming_a_completed_thread_returns_its_result_without_rerunning() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), _completion("done")]
    )
    store = InMemoryCheckpointer()
    lookup = Counter()
    agent = _agent(provider, _registry(lookup=lookup), checkpointer=store)
    first = await agent.run("go", thread_id="t1")

    again = await agent.resume("t1")

    assert again.content == first.content == "done"
    assert lookup.count == 1, "a completed thread must not re-run its tools"
    assert provider.calls and len(provider.calls) == 2, "and must not pay for the LLM again"
    assert [m.role for m in again.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_resuming_an_unknown_thread_raises_a_clear_error() -> None:
    store = InMemoryCheckpointer()
    agent = _agent(ScriptedProvider([]), checkpointer=store)
    await store.put(Checkpoint(thread_id="other"))

    with pytest.raises(UnknownThreadError) as exc:
        await agent.resume("nope")
    message = str(exc.value)
    assert "'nope'" in message
    assert "other" in message, "the error should list what is actually there"
    assert isinstance(exc.value, KeyError)


@pytest.mark.asyncio
async def test_resuming_a_failed_thread_refuses() -> None:
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), RuntimeError("boom")]
    )
    store = InMemoryCheckpointer()
    agent = _agent(provider, _registry(lookup=Counter()), checkpointer=store)
    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("go", thread_id="t1")

    with pytest.raises(RuntimeError, match="checkpointed as failed"):
        await agent.resume("t1")


@pytest.mark.asyncio
async def test_resume_without_a_checkpointer_is_rejected() -> None:
    agent = _agent(ScriptedProvider([]))
    with pytest.raises(ValueError, match="checkpointer"):
        await agent.resume("t1")


# ---------------------------------------------------------------------------
# Checkpointer implementations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_checkpointer_crud(backend: str, tmp_path: Path) -> None:
    store: Checkpointer = (
        InMemoryCheckpointer() if backend == "memory" else SqliteCheckpointer(tmp_path / "cp.db")
    )

    assert await store.get("t1") is None
    assert await store.list_threads() == []
    assert await store.delete("t1") is False

    await store.put(Checkpoint(thread_id="t1", status="running", tag="x"))
    got = await store.get("t1")
    assert got is not None and got.tag == "x"

    await store.put(Checkpoint(thread_id="t1", status="completed"))
    got = await store.get("t1")
    assert got is not None and got.status == "completed", "put must overwrite, not append"

    await store.put(Checkpoint(thread_id="t2"))
    assert sorted(await store.list_threads()) == ["t1", "t2"]

    assert await store.delete("t1") is True
    assert await store.get("t1") is None
    assert await store.list_threads() == ["t2"]


@pytest.mark.asyncio
async def test_in_memory_checkpointer_snapshots_what_it_is_given() -> None:
    """A stored reference would follow the agent's later mutations and lie about state."""
    store = InMemoryCheckpointer()
    checkpoint = Checkpoint(thread_id="t1", messages=[ChatMessage(role="user", content="a")])
    await store.put(checkpoint)

    checkpoint.messages.append(ChatMessage(role="assistant", content="b"))
    checkpoint.status = "completed"

    stored = await store.get("t1")
    assert stored is not None
    assert len(stored.messages) == 1
    assert stored.status == "running"


@pytest.mark.asyncio
async def test_resume_across_a_fresh_sqlite_checkpointer(tmp_path: Path) -> None:
    """Proves the state lives in the file, not in the checkpointer object's memory.

    A second ``SqliteCheckpointer`` on the same path, driving a second ``Agent``, is
    what a resume in a new process actually looks like.
    """
    db = tmp_path / "runs.db"
    provider = ScriptedProvider(
        [_completion("", [_call("lookup", "c1", x=1)]), RuntimeError("process died")]
    )
    lookup = Counter()
    agent = _agent(provider, _registry(lookup=lookup), checkpointer=SqliteCheckpointer(db))
    with pytest.raises(RuntimeError, match="process died"):
        await agent.run("go", thread_id="job-42")

    reopened = SqliteCheckpointer(db)
    stored = await reopened.get("job-42")
    assert stored is not None
    assert stored.status == "failed"
    stored.status = "running"
    stored.error = None
    await reopened.put(stored)

    provider.queue(_completion("finished"))
    fresh_agent = _agent(provider, _registry(lookup=lookup), checkpointer=SqliteCheckpointer(db))
    result = await fresh_agent.resume("job-42")

    assert result.content == "finished"
    assert lookup.count == 1, "the tool that already ran must not run again in the new process"


@pytest.mark.asyncio
async def test_sqlite_interrupt_survives_a_fresh_checkpointer(tmp_path: Path) -> None:
    """HITL must work across processes: the pending call lives in the file."""
    db = tmp_path / "runs.db"
    provider = ScriptedProvider([_completion("", [_call("unsafe_send", "c1", x=1)])])
    send = Counter()
    agent = _agent(
        provider,
        _registry(unsafe_send=send),
        checkpointer=SqliteCheckpointer(db),
        interrupt_before=["unsafe_send"],
    )
    paused = await agent.run("go", thread_id="job-7")
    assert paused.interrupted

    provider.queue(_completion("sent"))
    approver = _agent(
        provider,
        _registry(unsafe_send=send),
        checkpointer=SqliteCheckpointer(db),
        interrupt_before=["unsafe_send"],
    )
    result = await approver.resume("job-7", approve=True)

    assert result.content == "sent"
    assert send.count == 1


@pytest.mark.asyncio
async def test_sqlite_checkpointer_isolates_thread_ids(tmp_path: Path) -> None:
    """Concurrent writes to different threads on one file must not lose either."""
    store = SqliteCheckpointer(tmp_path / "runs.db")
    await asyncio.gather(*(store.put(Checkpoint(thread_id=f"t{i}", tag=str(i))) for i in range(20)))
    assert sorted(await store.list_threads()) == sorted(f"t{i}" for i in range(20))
    for i in range(20):
        got = await store.get(f"t{i}")
        assert got is not None and got.tag == str(i)


def test_sqlite_checkpointer_stamps_its_schema_version(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "runs.db"
    asyncio.run(SqliteCheckpointer(db).put(Checkpoint(thread_id="t1")))
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_sqlite_checkpointer_refuses_an_incompatible_schema(tmp_path: Path) -> None:
    """Never reset: these rows are the only record of which side effects already ran."""
    import sqlite3

    db = tmp_path / "runs.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY)")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(CheckpointSchemaMismatch) as exc:
        asyncio.run(SqliteCheckpointer(db).get("t1"))
    message = str(exc.value)
    assert str(SCHEMA_VERSION) in message
    assert "already happened" in message

    # The file is untouched, so the thread can still be finished by the version that
    # wrote it.
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 1
    finally:
        conn.close()


def test_checkpoint_round_trips_through_json() -> None:
    """Serialization is model_dump_json and nothing else; nothing may be lost."""
    original = Checkpoint(
        thread_id="t1",
        status="interrupted",
        messages=[
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content="", tool_calls=[_call("send", "c1", x=1)]),
        ],
        pending_call=_call("send", "c1", x=1),
        step_index=3,
        max_steps=6,
        tag="job",
    )
    assert Checkpoint.model_validate_json(original.model_dump_json()) == original


def test_a_custom_checkpointer_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryCheckpointer(), Checkpointer)
    assert isinstance(SqliteCheckpointer("x.db"), Checkpointer)
    assert not isinstance(object(), Checkpointer)


def test_agent_rejects_a_checkpointer_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="Checkpointer protocol"):
        Agent(
            llm=LLM(provider=ScriptedProvider([]), model="m", tracing=False),
            checkpointer=object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Checkpoint",
        "Checkpointer",
        "CheckpointStatus",
        "StepRecord",
        "InMemoryCheckpointer",
        "SqliteCheckpointer",
        "ResumeResolution",
        "CheckpointError",
        "CheckpointSchemaMismatch",
        "UnknownThreadError",
        "UnresolvedToolCallError",
    ],
)
def test_durability_symbols_are_public(name: str) -> None:
    import actants

    assert name in actants.__all__
    assert getattr(actants, name) is not None


@pytest.mark.parametrize(
    "name", ["CheckpointError", "CheckpointSchemaMismatch", "UnknownThreadError"]
)
def test_checkpoint_errors_join_the_hierarchy(name: str) -> None:
    import actants

    assert issubclass(getattr(actants, name), ActantsError)


def test_unresolved_tool_call_error_joins_the_hierarchy() -> None:
    assert issubclass(UnresolvedToolCallError, ActantsError)
    assert issubclass(UnresolvedToolCallError, RuntimeError)
