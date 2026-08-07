from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from actants.agents.events import (
    AgentRunCompleted,
    AgentStepCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)
from actants.agents.hooks import AgentHooks
from actants.agents.memory import ConversationMemory  # noqa: TC001 — runtime use
from actants.cost.tracker import CostTracker  # noqa: TC001 — runtime use in signatures
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
from actants.tools.base import serialize_tool_result
from actants.tools.registry import ToolRegistry

AgentEvent = (
    AgentTextDelta
    | AgentToolCallStarted
    | AgentToolCallCompleted
    | AgentStepCompleted
    | AgentRunCompleted
)

#: How concurrent ``run()`` calls on one Agent share its ConversationMemory.
#: See the :class:`Agent` docstring for the guarantee each one provides.
#:
#: Spelled as a ``Literal`` rather than an enum to match the rest of the public API
#: (``Role``, ``LogFormat``, ``LogLevel``): callers pass a plain string, and a type
#: checker rejects a typo at the call site instead of at runtime.
ConcurrencyMode = Literal["isolated", "serialized"]

#: Runtime mirror of :data:`ConcurrencyMode`, for the constructor check that catches
#: callers who are not running a type checker.
_CONCURRENCY_MODES: tuple[ConcurrencyMode, ...] = ("isolated", "serialized")


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
class AgentResult:
    final: CompletionResult
    steps: list[AgentStep]
    messages: list[ChatMessage]

    @property
    def content(self) -> str:
        return self.final.content


class Agent:
    """Stateful tool-calling agent.

    Wraps :class:`LLM` with conversation memory, tool registry, and lifecycle hooks.
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
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationMemory(system=system)
        self.hooks = hooks or AgentHooks()
        self.max_steps = max_steps
        self.concurrency: ConcurrencyMode = concurrency
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _turn(self, prompt: str) -> AsyncIterator[_Turn]:
        """Scope one run's conversation state; see the class docstring for the contract.

        Yields a :class:`_Turn` holding the working history for this run. In
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
    ) -> AgentResult:
        """Run one user turn; may dispatch tools across multiple LLM steps.

        Concurrency-safe: see the :class:`Agent` docstring for exactly what concurrent
        runs on one Agent are guaranteed to see.
        """
        steps: list[AgentStep] = []
        limit = max_steps if max_steps is not None else self.max_steps
        specs = self.tools.as_specs() if self.tools else None
        result: AgentResult | None = None

        try:
            async with self._turn(prompt) as turn:
                for i in range(limit):
                    msgs = turn.memory.messages()
                    if self.hooks.before_step is not None:
                        await self.hooks.before_step(i, msgs)

                    completion = await self.llm.complete(
                        msgs,
                        tools=specs,
                        use_cache=False,
                        tag=tag,
                    )

                    step = AgentStep(
                        index=i, completion=completion, tool_calls=completion.tool_calls
                    )
                    steps.append(step)

                    if self.hooks.after_step is not None:
                        await self.hooks.after_step(i, completion)

                    if not completion.tool_calls:
                        turn.add(ChatMessage(role="assistant", content=completion.content))
                        # Built here, returned after the turn commits: the caller's
                        # ``messages`` is this run's own history, not the agent's memory,
                        # which by then may also hold turns other runs committed.
                        result = AgentResult(
                            final=completion,
                            steps=steps,
                            messages=turn.memory.messages(),
                        )
                        break

                    turn.add(
                        ChatMessage(
                            role="assistant",
                            content=completion.content,
                            tool_calls=completion.tool_calls,
                        )
                    )
                    if self.tools is None:
                        raise RuntimeError(
                            "Model requested tool calls but no ToolRegistry was provided"
                        )
                    for call in completion.tool_calls:
                        result = await self.tools.call(call.name, **call.arguments)
                        payload = serialize_tool_result(result)
                        step.tool_results.append(payload)
                        if self.hooks.on_tool_call is not None:
                            await self.hooks.on_tool_call(call, result.value if result.ok else None)
                        turn.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
                else:
                    raise RuntimeError(
                        f"Agent exceeded max_steps={limit} without producing a final answer"
                    )
            assert result is not None  # the loop either sets it or raises
            return result
        except Exception as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise

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
        and the same concurrency guarantee applies — see the :class:`Agent` docstring.
        A stream that is abandoned part-way (the consumer stops iterating) commits
        nothing in the default ``isolated`` mode.
        """
        limit = max_steps if max_steps is not None else self.max_steps
        specs = self.tools.as_specs() if self.tools else None

        model = self.llm.settings.model
        cost_tracker = self.llm.cost_tracker

        try:
            async with self._turn(prompt) as turn:
                async for event in self._stream_turn(
                    turn, limit=limit, specs=specs, model=model, cost_tracker=cost_tracker, tag=tag
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
        cost_tracker: CostTracker | None,
        tag: str | None,
    ) -> AsyncIterator[AgentEvent]:
        """The step loop for :meth:`stream`, factored out so the turn scope wraps it."""
        for step_idx in range(limit):
            msgs = turn.memory.messages()
            if self.hooks.before_step is not None:
                await self.hooks.before_step(step_idx, msgs)

            step_text: list[str] = []
            step_tool_calls: list[ToolCall] = []
            last_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            last_cost = 0.0

            # Through the LLM layer, not the provider: this is what gives a streamed
            # run the same retry, tracing, and per-call override semantics as run().
            async for event in self.llm.stream_events(
                msgs,
                max_tokens=None,
                tools=specs,
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
            if cost_tracker is not None:
                cost_tracker.record(completion, tag=tag)

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
