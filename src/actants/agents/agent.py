from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Literal, NoReturn

from actants.agents.checkpoint import (
    RESUME_FAILED_ACKNOWLEDGED,
    Checkpoint,
    Checkpointer,  # noqa: TC001 — runtime use in the constructor's isinstance check
    CheckpointStatus,
    ResumeFailedAck,
    record_to_step,
    step_to_record,
)
from actants.agents.events import (
    AgentRunCompleted,
    AgentStepCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)
from actants.agents.hooks import AgentHooks
from actants.agents.memory import ConversationMemory  # noqa: TC001 — runtime use
from actants.errors import ActantsError, UnknownThreadError, UnresolvedToolCallError
from actants.llm.base import (
    ChatMessage,
    CompletionResult,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolSpec,  # noqa: TC001 — runtime use in signatures
    UsageDelta,
)
from actants.llm.client import LLM
from actants.tools.base import ToolError, serialize_tool_result
from actants.tools.registry import ToolRegistry

AgentEvent = (
    AgentTextDelta
    | AgentToolCallStarted
    | AgentToolCallCompleted
    | AgentStepCompleted
    | AgentRunCompleted
)

#: How concurrent ``run()`` calls on one Agent share its ConversationMemory.
#: See the `Agent` docstring for the guarantee each one provides.
#:
#: Spelled as a ``Literal`` rather than an enum to match the rest of the public API
#: (``Role``, ``LogFormat``, ``LogLevel``): callers pass a plain string, and a type
#: checker rejects a typo at the call site instead of at runtime.
ConcurrencyMode = Literal["isolated", "serialized"]

#: Runtime mirror of `ConcurrencyMode`, for the constructor check that catches
#: callers who are not running a type checker.
_CONCURRENCY_MODES: tuple[ConcurrencyMode, ...] = ("isolated", "serialized")

#: What `resume` should do about a tool call that was in flight when the
#: process died, for a tool declared ``idempotent=False``. See `resume`.
ResumeResolution = Literal["abort", "retry", "skip"]

#: Runtime mirror of `ResumeResolution`.
_RESUME_RESOLUTIONS: tuple[ResumeResolution, ...] = ("abort", "retry", "skip")

#: The tool result recorded for a call a human rejected at an ``interrupt_before`` pause,
#: and for one ``resolve="skip"`` declined to replay. Deliberately shaped like any other
#: failed tool result, so the model reacts to it the way it reacts to a tool error rather
#: than needing to understand actants' pause machinery.
_REJECTED_PAYLOAD = '{"error": "The tool call was rejected by a human reviewer and was not run."}'
_SKIPPED_PAYLOAD = (
    '{"error": "The tool call was interrupted by a crash and was not re-run, because the '
    'tool is not safe to repeat. Its result is unknown."}'
)


@dataclass
class _Turn:
    """The conversation state one run() is working against.

    ``memory`` is what the run reads and appends to — the agent's own memory in
    ``serialized`` mode, a private copy in ``isolated`` mode. ``pending`` collects the
    messages this turn produced so they can be committed to the agent's memory in one
    batch; it is None when the run is already writing through to the real memory.
    """

    memory: ConversationMemory
    pending: list[ChatMessage] | None

    def add(self, message: ChatMessage) -> None:
        self.memory.add(message)
        if self.pending is not None:
            self.pending.append(message)


@dataclass
class AgentStep:
    index: int
    completion: CompletionResult
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


@dataclass
class _RunState:
    """The bookkeeping one run carries, whether or not it is being checkpointed.

    ``thread_id`` is None for a non-durable run, and every checkpoint write is a no-op
    in that case — which is what keeps the un-threaded path identical to the code before
    durability existed.
    """

    thread_id: str | None
    tag: str | None
    max_steps: int
    steps: list[AgentStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    #: Failures this thread was resumed past, carried so later writes cannot erase them.
    prior_errors: list[str] = field(default_factory=list)


class _Interrupted(ActantsError):
    """Carries an interrupted run's result out through the turn scope.

    Private, and never reaches a caller: ``run``/``resume`` catch it and return
    `result`. It exists so a pause unwinds ``_turn`` instead of returning through
    it, leaving the half-written turn uncommitted. It still inherits ``ActantsError``, so
    the "every actants exception is catchable as one" invariant holds even for a class
    nobody is expected to catch.
    """

    def __init__(self, result: AgentResult) -> None:
        super().__init__("agent run interrupted")
        self.result = result


@dataclass
class AgentResult:
    """What one ``run()`` or ``resume()`` produced.

    A run that stopped at an ``interrupt_before`` tool sets `interrupted` and
    `pending_call` and leaves `final` holding the completion that *asked* for
    that call — the model's last word before it was paused. Call
    `resume` with the thread id to continue.
    """

    final: CompletionResult
    steps: list[AgentStep]
    messages: list[ChatMessage]
    #: True when the run stopped in front of an ``interrupt_before`` tool call rather
    #: than producing a final answer.
    interrupted: bool = False
    #: The call the run stopped in front of and did *not* dispatch. Set only when
    #: `interrupted`.
    pending_call: ToolCall | None = None
    #: The durable thread this run was checkpointed under, if any.
    thread_id: str | None = None

    @property
    def content(self) -> str:
        return self.final.content


class Agent:
    """Stateful tool-calling agent.

    Wraps `LLM` with conversation memory, tool registry, and lifecycle hooks.
    For one-shot tool loops without state, use ``LLM.run_agent`` directly.

    ``llm`` defaults to ``LLM()``, i.e. Ollama on localhost.

    Example::

        agent = Agent()  # local Ollama, no tools
        agent = Agent(llm=LLM(), tools=registry, system="You are a helpful assistant")
        result = await agent.run("what's the weather?")
        result2 = await agent.run("and tomorrow?")  # remembers context

    Concurrency guarantee
    ---------------------
    **Each ``run()`` / ``stream()`` sees a private copy of the conversation, and commits
    its turn back to the agent's memory atomically when it finishes.** Concurrent runs on
    one Agent therefore never observe each other's partial state: a run started while
    another is in flight is seeded from the memory as it stood at that moment, and no run
    can ever see a half-written turn — an assistant message without its tool results, or
    another run's user message interleaved into its own history.

    This is the ``concurrency="isolated"`` default. It makes N concurrent runs behave
    like N independent conversations that happen to share a starting point. Commit order
    is the order runs *finish*, so the agent's memory afterwards contains every turn,
    each one contiguous.

    Set ``concurrency="serialized"`` for the other reasonable contract: runs take an
    ``asyncio.Lock`` and execute one at a time, so each one sees every turn committed
    before it. Use this when later turns must build on earlier ones — a genuine
    multi-turn conversation driven from several tasks. It trades throughput for
    ordering; runs queue rather than overlap.

    Neither mode makes ``Agent`` safe to share across OS threads or processes; the lock
    is an ``asyncio`` lock and the memory is a plain list. One Agent per event loop.

    A single sequential caller — the common case — is unaffected by this setting::

        result = await agent.run("first")
        result2 = await agent.run("second")   # sees "first" either way

    Durability
    ----------
    Pass a ``checkpointer`` and a ``thread_id`` to make a run resumable::

        agent = Agent(llm=LLM(), tools=tools, checkpointer=SqliteCheckpointer("runs.db"))
        result = await agent.run("book the flight", thread_id="job-42")
        # ...process dies...
        result = await agent.resume("job-42")   # picks up where it left off

    ``interrupt_before=["send_email"]`` pauses the run in front of those tools instead of
    dispatching them; `resume` with ``approve=True`` or ``approve=False`` decides.

    Both default to None, and a run without a ``thread_id`` never touches the
    checkpointer — durability is opt-in per run, not a mode the agent is in.
    """

    def __init__(
        self,
        *,
        llm: LLM | None = None,
        tools: ToolRegistry | None = None,
        system: str | None = None,
        memory: ConversationMemory | None = None,
        hooks: AgentHooks | None = None,
        max_steps: int = 6,
        concurrency: ConcurrencyMode = "isolated",
        checkpointer: Checkpointer | None = None,
        interrupt_before: Iterable[str] | None = None,
    ) -> None:
        if llm is None:
            llm = LLM()
        if not isinstance(llm, LLM):
            raise TypeError(
                f"llm must be an LLM instance, got {type(llm).__name__!r}. "
                "Pass Agent(llm=LLM()) for the local Ollama default, or "
                "Agent(llm=LLM(provider='openai', model='gpt-4o')) for a hosted provider."
            )
        if tools is not None and not isinstance(tools, ToolRegistry):
            raise TypeError(
                f"tools must be a ToolRegistry, got {type(tools).__name__!r}. "
                "Build one with:\n"
                "    registry = ToolRegistry()\n"
                "    registry.register_function('add', 'Add two integers', add)\n"
                "    Agent(llm=LLM(), tools=registry)"
            )
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a string, got {type(system).__name__!r}. "
                "Example: Agent(llm=LLM(), system='You are a helpful assistant')."
            )
        if memory is not None and not isinstance(memory, ConversationMemory):
            raise TypeError(
                f"memory must be a ConversationMemory, got {type(memory).__name__!r}. "
                "Example: Agent(llm=LLM(), memory=ConversationMemory(max_messages=20))."
            )
        if hooks is not None and not isinstance(hooks, AgentHooks):
            raise TypeError(
                f"hooks must be an AgentHooks, got {type(hooks).__name__!r}. "
                "Example: Agent(llm=LLM(), hooks=AgentHooks(on_tool_call=my_callback))."
            )
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(
                f"max_steps must be an integer >= 1, got {max_steps!r}. "
                "It caps how many LLM round-trips one run() may take."
            )
        if memory is not None and system is not None:
            raise ValueError(
                "Pass either system= or memory=, not both — a ConversationMemory already "
                "carries its own system prompt. Use "
                "ConversationMemory(system='...') and pass it as memory=."
            )
        if concurrency not in _CONCURRENCY_MODES:
            raise ValueError(
                f"concurrency must be one of {list(_CONCURRENCY_MODES)}, got "
                f"{concurrency!r}. 'isolated' (the default) gives each run() a private "
                "copy of the conversation, committed back when it finishes. "
                "'serialized' makes concurrent runs queue on a lock so each sees every "
                "earlier turn. Example: Agent(llm=LLM(), concurrency='serialized')."
            )
        if checkpointer is not None and not isinstance(checkpointer, Checkpointer):
            raise TypeError(
                f"checkpointer must implement the Checkpointer protocol (put/get/"
                f"list_threads/delete), got {type(checkpointer).__name__!r}. "
                "Example: Agent(llm=LLM(), checkpointer=SqliteCheckpointer('runs.db'))."
            )
        if interrupt_before is not None and isinstance(interrupt_before, str):
            raise TypeError(
                "interrupt_before must be a collection of tool names, not a single "
                f"string ({interrupt_before!r} would be read one character at a time). "
                f"Use interrupt_before=[{interrupt_before!r}]."
            )
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationMemory(system=system)
        self.hooks = hooks or AgentHooks()
        self.max_steps = max_steps
        self.concurrency: ConcurrencyMode = concurrency
        self.checkpointer = checkpointer
        self.interrupt_before: frozenset[str] = frozenset(interrupt_before or ())
        self._lock = asyncio.Lock()
        self._resume_locks: dict[str, asyncio.Lock] = {}

    @contextlib.asynccontextmanager
    async def _turn(self, prompt: str) -> AsyncIterator[_Turn]:
        """Scope one run's conversation state; see the class docstring for the contract.

        Yields a `_Turn` holding the working history for this run. In
        ``isolated`` mode that history is a private copy seeded from the agent's memory,
        and the messages the run appends are committed back in one batch on success —
        so no other run can ever observe a partially-written turn. In ``serialized``
        mode the run holds the agent's lock and writes straight through.

        A run that raises commits nothing in isolated mode: a failed turn leaves the
        conversation as it was rather than stranding a user message with no answer.
        """
        if self.concurrency == "serialized":
            async with self._lock:
                self.memory.add_user(prompt)
                yield _Turn(memory=self.memory, pending=None)
            return

        # isolated: snapshot, work privately, commit on success.
        working = ConversationMemory()
        working.extend(self.memory.messages())
        working.add_user(prompt)
        pending: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]
        turn = _Turn(memory=working, pending=pending)
        yield turn
        # Reached only when the body did not raise.
        async with self._lock:
            self.memory.extend(pending)

    async def run(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        tag: str | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """Run one user turn; may dispatch tools across multiple LLM steps.

        Concurrency-safe: see the `Agent` docstring for exactly what concurrent
        runs on one Agent are guaranteed to see.

        Pass ``thread_id`` — with a ``checkpointer`` on the agent — to make the run
        durable. Its state is persisted after every LLM completion and after *each*
        individual tool result, so `resume` can continue it after a crash without
        re-running the tools that already finished. ``thread_id`` without a checkpointer
        is a ``ValueError``; a checkpointer without a ``thread_id`` simply persists
        nothing, which is what keeps un-threaded runs exactly as they were.
        """
        if thread_id is not None and self.checkpointer is None:
            raise ValueError(
                f"run(thread_id={thread_id!r}) needs a checkpointer to persist to, and "
                "this Agent has none. Construct it with "
                "Agent(..., checkpointer=SqliteCheckpointer('runs.db')), or drop "
                "thread_id for a non-durable run."
            )

        limit = max_steps if max_steps is not None else self.max_steps
        state = _RunState(thread_id=thread_id, tag=tag, max_steps=limit)

        try:
            async with self._turn(prompt) as turn:
                result = await self._run_steps(turn, state, start_step=0)
        except _Interrupted as pause:
            # Deliberately unwinds through ``_turn`` rather than returning through it: a
            # paused turn is half-written (an assistant message whose tool results have
            # not happened yet) and must not be committed to the agent's memory, exactly
            # like a failed one. The resumed run commits the whole turn when it finishes.
            return pause.result
        except Exception as exc:
            await self._checkpoint_failure(state, exc)
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise
        return result

    async def resume(
        self,
        thread_id: str,
        *,
        approve: bool | None = None,
        max_steps: int | None = None,
        resolve: ResumeResolution = "abort",
        resume_failed: ResumeFailedAck | None = None,
    ) -> AgentResult:
        """Continue a checkpointed run from where it stopped.

        **The guarantee.** Every tool call whose result was recorded before the crash is
        replayed from the checkpoint and never dispatched again — resume is at-most-once
        for those. The one call that was *in flight* when the process died is the
        irreducible ambiguity: actants cannot know whether its side effect landed,
        because the process died before it could say so. For a tool declared
        ``idempotent=True`` (the default) that call is re-dispatched, making resume
        at-least-once for it; for ``idempotent=False`` it is surfaced as
        `UnresolvedToolCallError` instead of being guessed at. A
        tool this Agent's registry no longer has — renamed or removed since the run
        started — counts as non-idempotent, because an unknown tool is precisely the case
        where actants cannot establish that repeating it is safe.

        ``approve`` answers a run paused by ``interrupt_before``: ``True`` dispatches the
        pending call and continues, ``False`` appends a tool result saying the call was
        rejected, so the model gets to react to the refusal rather than the run dying.

        ``resolve`` decides what to do with an in-flight non-idempotent call, and is
        ignored otherwise: ``"abort"`` (the default) raises; ``"retry"`` dispatches it
        anyway, for when the caller has established it did not run; ``"skip"`` records
        that it was not run and lets the model continue without it.

        Resuming a thread that already completed returns its stored result without
        re-running anything. Resuming one that failed re-raises a description of the
        original failure, unless ``resume_failed=RESUME_FAILED_ACKNOWLEDGED`` is passed:
        an unknown failure may have half-run, so continuing past one is a judgement only a
        human has the context to make, and the opt-in is spelled as a sentence so it
        cannot arrive from a wrapper forwarding flags. It does **not** loosen anything
        above it — a thread that died mid-call still goes through ``resolve``, so a
        non-idempotent call still raises `UnresolvedToolCallError`.
        An unknown ``thread_id`` raises `UnknownThreadError`.

        Concurrent resumes of one ``thread_id`` **within this process** are serialized on
        a per-thread lock, so the read-decide-dispatch sequence cannot interleave and the
        at-most-once guarantee above holds for them: the second resume runs after the
        first has committed, sees a completed thread, and returns its stored result
        without dispatching anything. Two *processes* resuming the same ``thread_id``
        concurrently remains undefined — actants does not lock a thread across processes,
        and no in-process lock can.
        """
        async with self._resume_guard(thread_id):
            return await self._resume_locked(
                thread_id,
                approve=approve,
                max_steps=max_steps,
                resolve=resolve,
                resume_failed=resume_failed,
            )

    def _resume_guard(self, thread_id: str) -> asyncio.Lock:
        """The lock serializing resumes of one thread inside this process.

        Per thread_id rather than one agent-wide lock so unrelated threads still resume
        concurrently. Created on demand and kept: a lock is a few dozen bytes, and the
        alternative — reference-counting them to delete on release — is a race in itself.
        """
        lock = self._resume_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._resume_locks[thread_id] = lock
        return lock

    async def _resume_locked(
        self,
        thread_id: str,
        *,
        approve: bool | None,
        max_steps: int | None,
        resolve: ResumeResolution,
        resume_failed: ResumeFailedAck | None = None,
    ) -> AgentResult:
        """The body of `resume`, run while holding that thread's resume lock."""
        if self.checkpointer is None:
            raise ValueError(
                f"resume({thread_id!r}) needs a checkpointer to read from, and this "
                "Agent has none. Resume on an Agent constructed with the same "
                "checkpointer the run was started with."
            )
        if resolve not in _RESUME_RESOLUTIONS:
            raise ValueError(
                f"resolve must be one of {list(_RESUME_RESOLUTIONS)}, got {resolve!r}. "
                "'abort' (the default) raises UnresolvedToolCallError for an in-flight "
                "non-idempotent call, 'retry' dispatches it anyway, 'skip' records that "
                "it was not run."
            )
        if resume_failed is not None and resume_failed != RESUME_FAILED_ACKNOWLEDGED:
            raise ValueError(
                f"resume_failed must be exactly {RESUME_FAILED_ACKNOWLEDGED!r} (importable "
                f"as RESUME_FAILED_ACKNOWLEDGED), got {resume_failed!r}. It is spelled out "
                "so resuming a failure is never something a caller does by accident."
            )

        checkpoint = await self.checkpointer.get(thread_id)
        if checkpoint is None:
            known = await self.checkpointer.list_threads()
            listed = ", ".join(sorted(known)[:10]) or "<none>"
            raise UnknownThreadError(
                f"No checkpoint for thread_id {thread_id!r}. Either it never ran under "
                f"this checkpointer, or its state was deleted. Known threads: {listed}."
            )
        if checkpoint.status == "completed":
            return self._result_from_checkpoint(checkpoint)
        if checkpoint.status == "failed" and resume_failed is None:
            raise RuntimeError(
                f"Thread {thread_id!r} is checkpointed as failed and cannot be resumed: "
                f"{checkpoint.error}. Start a new run, or delete the thread first. If you "
                "have established that continuing is safe, resume with "
                f"resume_failed={RESUME_FAILED_ACKNOWLEDGED!r} — the completed steps are "
                "still in the checkpoint and are replayed, not re-run."
            )
        if checkpoint.status == "interrupted" and approve is None:
            assert checkpoint.pending_call is not None  # invariant of "interrupted"
            raise ValueError(
                f"Thread {thread_id!r} is paused before tool "
                f"{checkpoint.pending_call.name!r} and needs a decision. Call "
                f"resume({thread_id!r}, approve=True) to dispatch it, or approve=False "
                "to reject it and let the model respond to the refusal."
            )

        limit = max_steps if max_steps is not None else (checkpoint.max_steps or self.max_steps)
        # The failure being resumed past moves into the history now, so it survives every
        # write this run makes — including a second failure overwriting ``error``.
        history = list(checkpoint.prior_errors)
        if checkpoint.status == "failed" and checkpoint.error is not None:
            history.append(checkpoint.error)
        state = _RunState(
            thread_id=thread_id,
            tag=checkpoint.tag,
            max_steps=limit,
            steps=[record_to_step(r) for r in checkpoint.steps],
            created_at=checkpoint.created_at,
            prior_errors=history,
        )

        turn = self._turn_from_checkpoint(checkpoint)
        try:
            start_step = await self._resolve_pending_step(
                turn, state, checkpoint, approve=approve, resolve=resolve
            )
            result = await self._run_steps(turn, state, start_step=start_step)
        except _Interrupted as pause:
            return pause.result
        except Exception as exc:
            await self._checkpoint_failure(state, exc)
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise
        await self._commit_turn(turn)
        return result

    async def _run_steps(self, turn: _Turn, state: _RunState, *, start_step: int) -> AgentResult:
        """The step loop shared by `run` and `resume`.

        ``start_step`` is the next step needing an LLM call; a fresh run passes 0 and the
        loop is the original one. A resumed run's half-finished step has already been
        completed by `_resolve_pending_step` before this is entered, so every step
        this loop sees starts at its LLM call.
        """
        specs = self.tools.as_specs() if self.tools else None
        limit = state.max_steps

        for i in range(start_step, limit):
            msgs = turn.memory.messages()
            if self.hooks.before_step is not None:
                await self.hooks.before_step(i, msgs)

            completion = await self.llm.complete(
                msgs,
                tools=specs,
                use_cache=False,
                tag=state.tag,
            )

            step = AgentStep(index=i, completion=completion, tool_calls=completion.tool_calls)
            state.steps.append(step)

            if self.hooks.after_step is not None:
                await self.hooks.after_step(i, completion)

            if not completion.tool_calls:
                turn.add(ChatMessage(role="assistant", content=completion.content))
                # Built here, returned after the turn commits: the caller's
                # ``messages`` is this run's own history, not the agent's memory,
                # which by then may also hold turns other runs committed.
                result = AgentResult(
                    final=completion,
                    steps=state.steps,
                    messages=turn.memory.messages(),
                    thread_id=state.thread_id,
                )
                await self._checkpoint(turn, state, status="completed", step_index=i)
                return result

            turn.add(
                ChatMessage(
                    role="assistant",
                    content=completion.content,
                    tool_calls=completion.tool_calls,
                )
            )
            # Before any tool runs: a crash between the LLM call and dispatch would
            # otherwise pay for the completion twice, and the replayed model might well
            # ask for different calls the second time.
            await self._checkpoint(turn, state, status="running", step_index=i)
            await self._dispatch_calls(turn, state, step, completion.tool_calls, start_index=0)

        raise RuntimeError(f"Agent exceeded max_steps={limit} without producing a final answer")

    async def _dispatch_calls(
        self,
        turn: _Turn,
        state: _RunState,
        step: AgentStep,
        calls: list[ToolCall],
        *,
        start_index: int,
    ) -> None:
        """Run one step's tool calls, checkpointing after each result lands.

        Raises `_Interrupted` if a call named in ``interrupt_before`` is reached.
        ``start_index`` skips the calls a resumed step already has results for.
        """
        if self.tools is None:
            raise RuntimeError("Model requested tool calls but no ToolRegistry was provided")
        for call in calls[start_index:]:
            if call.name in self.interrupt_before:
                await self._interrupt(turn, state, step, call)

            # Deliberately not named `result`: that name holds this run's AgentResult.
            tool_result = await self.tools.call(call.name, **call.arguments)
            payload = serialize_tool_result(tool_result)
            step.tool_results.append(payload)
            if self.hooks.on_tool_call is not None:
                await self.hooks.on_tool_call(call, tool_result.value if tool_result.ok else None)
            turn.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
            # After the result is in the history, so the durable record says this exact
            # side effect happened and resume must not repeat it.
            await self._checkpoint(turn, state, status="running", step_index=step.index)
        return None

    async def _interrupt(
        self,
        turn: _Turn,
        state: _RunState,
        step: AgentStep,
        call: ToolCall,
    ) -> NoReturn:
        """Persist the paused run and unwind with the pending call."""
        if self.checkpointer is None or state.thread_id is None:
            raise ValueError(
                f"Tool {call.name!r} is in interrupt_before, but this run has no "
                "checkpointer and thread_id to persist the pause to — there would be "
                "nothing to resume from. Give the Agent a checkpointer and call "
                "run(prompt, thread_id=...)."
            )
        await self._checkpoint(
            turn, state, status="interrupted", step_index=step.index, pending_call=call
        )
        raise _Interrupted(
            AgentResult(
                final=step.completion,
                steps=state.steps,
                messages=turn.memory.messages(),
                interrupted=True,
                pending_call=call,
                thread_id=state.thread_id,
            )
        )

    async def _checkpoint(
        self,
        turn: _Turn,
        state: _RunState,
        *,
        status: CheckpointStatus,
        step_index: int,
        pending_call: ToolCall | None = None,
    ) -> None:
        """Persist the run's state, or do nothing if this run is not durable."""
        if self.checkpointer is None or state.thread_id is None:
            return
        await self.checkpointer.put(
            Checkpoint(
                thread_id=state.thread_id,
                status=status,
                messages=turn.memory.messages(),
                steps=[step_to_record(s) for s in state.steps],
                step_index=step_index,
                max_steps=state.max_steps,
                pending_call=pending_call,
                tag=state.tag,
                created_at=state.created_at,
                prior_errors=list(state.prior_errors),
            )
        )

    async def _checkpoint_failure(self, state: _RunState, exc: BaseException) -> None:
        """Mark the thread failed, preserving whatever was already durably recorded.

        Reads the stored checkpoint and flips its status rather than writing fresh state:
        the failure may have come from a torn turn whose in-memory history is not worth
        trusting, and the last good checkpoint is exactly the one worth keeping. A store
        that itself fails here is swallowed — the original exception is the one the
        caller needs to see.
        """
        if self.checkpointer is None or state.thread_id is None:
            return
        with contextlib.suppress(Exception):
            stored = await self.checkpointer.get(state.thread_id)
            if stored is None:
                return
            stored.status = "failed"
            stored.error = f"{type(exc).__name__}: {exc}"
            # From the run, not the stored copy: a resume that fails before writing
            # anything would otherwise leave the failure it was resumed past unrecorded.
            stored.prior_errors = list(state.prior_errors)
            stored.updated_at = time.time()
            await self.checkpointer.put(stored)

    def _turn_from_checkpoint(self, checkpoint: Checkpoint) -> _Turn:
        """Rebuild the working history of a resumed run.

        Always a private ``_Turn`` with a pending list, even in ``serialized`` mode: the
        checkpoint holds the whole turn including the user message, so writing it
        straight through to the agent's memory would duplicate whatever the original
        run had already committed there.
        """
        working = ConversationMemory()
        working.extend(checkpoint.messages)
        return _Turn(memory=working, pending=list(checkpoint.messages))

    async def _commit_turn(self, turn: _Turn) -> None:
        """Publish a resumed turn's messages to the agent's memory."""
        if turn.pending is None:
            return
        async with self._lock:
            self.memory.extend(turn.pending)

    def _result_from_checkpoint(self, checkpoint: Checkpoint) -> AgentResult:
        """Rebuild the result of a thread that already finished, without re-running it."""
        steps = [record_to_step(r) for r in checkpoint.steps]
        if not steps:
            raise RuntimeError(
                f"Thread {checkpoint.thread_id!r} is marked completed but has no recorded "
                "steps, so its result cannot be reconstructed. The checkpoint store may "
                "have been written by hand or truncated."
            )
        return AgentResult(
            final=steps[-1].completion,
            steps=steps,
            messages=list(checkpoint.messages),
            thread_id=checkpoint.thread_id,
        )

    async def _resolve_pending_step(
        self,
        turn: _Turn,
        state: _RunState,
        checkpoint: Checkpoint,
        *,
        approve: bool | None,
        resolve: ResumeResolution,
    ) -> int:
        """Finish the step the crash or pause landed in the middle of.

        Returns the step index `_run_steps` should start its next LLM call at. The
        half-done step is completed *here* rather than by re-entering the loop, because
        re-entering would issue a fresh completion for a step whose completion is already
        recorded — paying for the LLM twice and letting the model contradict the tool
        calls its own earlier answer asked for.
        """
        if not state.steps:
            return 0

        step = state.steps[-1]
        # A step whose completion asked for no tools is a finished step; the crash
        # happened after it, so the next LLM call is what resumes.
        if not step.tool_calls:
            return step.index + 1

        if checkpoint.status == "interrupted":
            assert checkpoint.pending_call is not None
            await self._resolve_interrupt(
                turn, state, step, checkpoint.pending_call, approve=bool(approve)
            )
            done = len(step.tool_results)
        else:
            # Only a "running" checkpoint has an ambiguous call: the process died with
            # one in flight. A pause is a clean stop — the calls after the pending one
            # provably never started, so they are dispatched normally rather than being
            # run through the idempotency question.
            done = await self._resolve_in_flight(
                turn, state, step, len(step.tool_results), resolve=resolve
            )
        if done < len(step.tool_calls):
            await self._dispatch_calls(turn, state, step, step.tool_calls, start_index=done)
        return step.index + 1

    async def _resolve_interrupt(
        self,
        turn: _Turn,
        state: _RunState,
        step: AgentStep,
        call: ToolCall,
        *,
        approve: bool,
    ) -> None:
        """Dispatch or reject the call an ``interrupt_before`` pause stopped in front of."""
        if not approve:
            step.tool_results.append(_REJECTED_PAYLOAD)
            turn.add(ChatMessage(role="tool", content=_REJECTED_PAYLOAD, tool_call_id=call.id))
            await self._checkpoint(turn, state, status="running", step_index=step.index)
            return
        if self.tools is None:
            raise RuntimeError("Model requested tool calls but no ToolRegistry was provided")
        tool_result = await self.tools.call(call.name, **call.arguments)
        payload = serialize_tool_result(tool_result)
        step.tool_results.append(payload)
        if self.hooks.on_tool_call is not None:
            await self.hooks.on_tool_call(call, tool_result.value if tool_result.ok else None)
        turn.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
        await self._checkpoint(turn, state, status="running", step_index=step.index)

    async def _resolve_in_flight(
        self,
        turn: _Turn,
        state: _RunState,
        step: AgentStep,
        done: int,
        *,
        resolve: ResumeResolution,
    ) -> int:
        """Decide the fate of the call that was executing when the process died.

        Only that one call is ambiguous — every call before it has a recorded result and
        is replayed, never re-dispatched. An idempotent tool is simply left to the normal
        loop to re-dispatch; a non-idempotent one — and a tool the registry no longer
        has — goes through ``resolve``.

        Returns the index the dispatch loop should start from.
        """
        if done >= len(step.tool_calls):
            return done
        call = step.tool_calls[done]
        # A tool missing from the registry — renamed, removed, or a registry the resuming
        # process assembled differently — is treated as NON-idempotent. It is exactly the
        # case where actants cannot know the side effect was safe to repeat, and the
        # cheerful default would otherwise re-dispatch it into `tools.call`, which reports
        # "Unknown tool" and thereby tells the model the side effect did not happen.
        idempotent = False
        known = False
        if self.tools is not None:
            with contextlib.suppress(ToolError):
                tool = self.tools.get(call.name)
                idempotent, known = tool.idempotent, True
        if idempotent or resolve == "retry":
            return done

        if resolve == "skip":
            step.tool_results.append(_SKIPPED_PAYLOAD)
            turn.add(ChatMessage(role="tool", content=_SKIPPED_PAYLOAD, tool_call_id=call.id))
            await self._checkpoint(turn, state, status="running", step_index=step.index)
            return done + 1

        why = (
            "which is registered idempotent=False"
            if known
            else "which this Agent's registry no longer has, so actants cannot tell "
            "whether repeating it is safe (a renamed or removed tool is treated as "
            "non-idempotent)"
        )
        raise UnresolvedToolCallError(
            f"Thread {state.thread_id!r} died while calling tool {call.name!r} "
            f"(call id {call.id!r}), {why}. actants "
            "cannot know whether that side effect happened, and will not guess. Check "
            "whether it ran — the call id is stable, so a vendor-side idempotency key or "
            "an outbox can be looked up by it — then resume with resolve='retry' to run "
            "it, or resolve='skip' to tell the model it did not run.",
            thread_id=state.thread_id or "",
            call=call,
        )

    async def stream(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        tag: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run one user turn, yielding lifecycle events as they happen.

        Yields, in order across N steps:
          - ``AgentTextDelta(text, step)`` — each model text chunk
          - ``AgentToolCallStarted(call, step)`` — when a tool call is dispatched
          - ``AgentToolCallCompleted(call, value, ok, step)`` — when it returns
          - ``AgentStepCompleted(step, completion)`` — end of one LLM call
          - ``AgentRunCompleted(content, final)`` — final answer (terminal)

        At the end of streaming, the agent state matches what ``run()`` would produce,
        and the same concurrency guarantee applies — see the `Agent` docstring.
        A stream that is abandoned part-way (the consumer stops iterating) commits
        nothing in the default ``isolated`` mode.
        """
        limit = max_steps if max_steps is not None else self.max_steps
        specs = self.tools.as_specs() if self.tools else None

        model = self.llm.settings.model

        try:
            async with self._turn(prompt) as turn:
                async for event in self._stream_turn(
                    turn, limit=limit, specs=specs, model=model, tag=tag
                ):
                    yield event
        except Exception as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise

    async def _stream_turn(
        self,
        turn: _Turn,
        *,
        limit: int,
        specs: list[ToolSpec] | None,
        model: str,
        tag: str | None,
    ) -> AsyncIterator[AgentEvent]:
        """The step loop for `stream`, factored out so the turn scope wraps it."""
        for step_idx in range(limit):
            msgs = turn.memory.messages()
            if self.hooks.before_step is not None:
                await self.hooks.before_step(step_idx, msgs)

            step_text: list[str] = []
            step_tool_calls: list[ToolCall] = []
            last_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            last_cost = 0.0

            # Through the LLM layer, not the provider: this is what gives a streamed
            # run the same retry, tracing, cost-tracking, and per-call override semantics
            # as run(). ``tag`` goes with it, so a tagged streamed agent run is
            # attributed exactly as a tagged run() is — and is recorded *there*, once,
            # rather than again below.
            async for event in self.llm.stream_events(
                msgs,
                max_tokens=None,
                tools=specs,
                tag=tag,
            ):
                if isinstance(event, TextDelta):
                    step_text.append(event.text)
                    yield AgentTextDelta(text=event.text, step=step_idx)
                elif isinstance(event, ToolCallDelta):
                    step_tool_calls.append(event.tool_call)
                elif isinstance(event, UsageDelta):
                    last_usage = event.usage
                    # Providers price the request themselves; dropping this field
                    # made every streamed run report $0.00.
                    last_cost = event.cost_usd

            completion = CompletionResult(
                content="".join(step_text),
                model=model,
                provider=self.llm.provider.name,
                usage=last_usage,
                cost_usd=last_cost,
                tool_calls=step_tool_calls,
            )
            # No cost_tracker.record() here: LLM.stream_events already recorded this
            # step's UsageDelta under ``tag``. Recording again would double every
            # streamed agent run's reported spend.

            if self.hooks.after_step is not None:
                await self.hooks.after_step(step_idx, completion)
            yield AgentStepCompleted(step=step_idx, completion=completion)

            if not step_tool_calls:
                turn.add(ChatMessage(role="assistant", content=completion.content))
                yield AgentRunCompleted(content=completion.content, final=completion)
                return

            turn.add(
                ChatMessage(
                    role="assistant",
                    content=completion.content,
                    tool_calls=step_tool_calls,
                )
            )
            if self.tools is None:
                raise RuntimeError("Model requested tool calls but no ToolRegistry was provided")
            for call in step_tool_calls:
                yield AgentToolCallStarted(call=call, step=step_idx)
                result = await self.tools.call(call.name, **call.arguments)
                payload = serialize_tool_result(result)
                yield AgentToolCallCompleted(
                    call=call,
                    value=result.value if result.ok else result.error,
                    ok=result.ok,
                    step=step_idx,
                )
                if self.hooks.on_tool_call is not None:
                    await self.hooks.on_tool_call(call, result.value if result.ok else None)
                turn.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))

        raise RuntimeError(f"Agent stream exceeded max_steps={limit} without a final answer")

    def reset(self, *, keep_system: bool = True) -> None:
        self.memory.reset(keep_system=keep_system)
