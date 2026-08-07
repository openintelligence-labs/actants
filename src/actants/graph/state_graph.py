"""Typed, durable state graphs for workflows that branch and loop.

:class:`~actants.agents.agent.Agent` is a linear loop: think, call tools, repeat until
done. A graph is for the shape that loop cannot express — a router that picks one of
three branches, a critic that sends work back for another pass, a pipeline whose stages
each need their own prompt.

The state is a pydantic model you define. Nodes receive it and return a partial update;
edges say what runs next. Both are typed against your model, so a node that reads a field
your state does not have is a type error rather than a runtime surprise::

    class State(BaseModel):
        question: str
        draft: str = ""
        critiques: Annotated[list[str], Append] = Field(default_factory=list)

    async def write(state: State) -> dict[str, Any]:
        return {"draft": f"answer to {state.question}"}

    async def critique(state: State) -> dict[str, Any]:
        return {"critiques": ["needs detail"]}

    def good_enough(state: State) -> str:
        return END if len(state.critiques) >= 2 else "write"

    graph = StateGraph(State)
    graph.add_node("write", write)
    graph.add_node("critique", critique)
    graph.set_entry_point("write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges("critique", good_enough, {"write": "write", END: END})

    compiled = graph.compile(checkpointer=SqliteCheckpointer("runs.db"))
    final = await compiled.invoke(State(question="why?"), thread_id="job-1")

Durability
----------
Pass a ``checkpointer`` to :meth:`StateGraph.compile` and a ``thread_id`` to
:meth:`CompiledGraph.invoke`, and the state is persisted after *every* node completes.
:meth:`CompiledGraph.resume` continues from the last completed node, and a node that
already ran is never re-run — the same at-most-once guarantee
:meth:`~actants.agents.agent.Agent.resume` gives for completed tool calls, applied at
node granularity.

This reuses the :class:`~actants.agents.checkpoint.Checkpointer` protocol the agent
uses, so one store — and one SQLite file — holds both agent runs and graph runs.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from actants.agents.checkpoint import (
    Checkpoint,
    Checkpointer,  # noqa: TC001 — runtime use in the constructor's isinstance check
    CheckpointStatus,
)
from actants.errors import GraphRecursionError, GraphValidationError, UnknownThreadError
from actants.graph.events import (
    GraphCompleted,
    GraphEvent,
    GraphInterrupted,
    GraphNodeCompleted,
    GraphNodeStarted,
)
from actants.graph.state import (
    END,
    append_fields,
    merge_update,
    state_from_json,
    state_to_json,
    validate_append_fields,
)
from actants.llm.base import ChatMessage, Role

#: A node: async, takes the state, returns an update. The type parameter is what makes a
#: node written against the wrong state model a type error at ``add_node``.
#:
#: The three-way return union lets a node be written in whichever style fits: a dict of
#: changed fields is the common case; returning the state object suits a node that
#: rebuilds it wholesale; None suits a node that only performs a side effect.
type NodeFn[StateT: BaseModel] = Callable[[StateT], Awaitable[dict[str, Any] | StateT | None]]

#: A router: sync, takes the state, returns a key into the mapping given alongside it.
#:
#: Sync deliberately — a router that needs to await something is doing work, and that
#: work belongs in a node where it is checkpointed. A router runs *after* its node's
#: state is durably recorded and must be cheap enough to simply re-run on resume.
type RouterFn[StateT: BaseModel] = Callable[[StateT], str]

#: Tag written into every graph checkpoint, so a store shared with agent runs can tell
#: the two apart — and so resuming a graph thread with ``Agent.resume`` fails loudly
#: instead of misreading the payload.
GRAPH_TAG = "actants.graph"

#: Role of the single synthetic message carrying serialized graph state.
#:
#: Graph state rides in ``Checkpoint.messages`` because that is the field the existing
#: schema already persists, which is what lets graphs reuse the agent's checkpointer,
#: its SQLite layout, and its cross-process guarantee without a second store or a schema
#: bump. "system" because the payload is framework bookkeeping, not conversation.
_STATE_ROLE: Role = "system"


@dataclass
class _Node[StateT: BaseModel]:
    """One registered node and the edges leaving it.

    ``static_target`` and ``router`` are mutually exclusive; ``compile`` enforces that a
    node has at most one of them.
    """

    name: str
    fn: NodeFn[StateT]
    static_target: str | None = None
    router: RouterFn[StateT] | None = None
    routes: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphResult[StateT: BaseModel]:
    """What one ``invoke()`` or ``resume()`` produced.

    A run that stopped in front of an ``interrupt_before`` node sets :attr:`interrupted`
    and :attr:`pending_node`, and :attr:`state` holds the state as it stood *before* that
    node ran. Deliberately the same shape and vocabulary as
    :class:`~actants.agents.agent.AgentResult`, so code that handles a paused agent
    handles a paused graph the same way.
    """

    state: StateT
    #: True when the run stopped in front of an ``interrupt_before`` node rather than
    #: reaching END.
    interrupted: bool = False
    #: The node the run stopped in front of and did *not* run. Set only when
    #: :attr:`interrupted`.
    pending_node: str | None = None
    #: The durable thread this run was checkpointed under, if any.
    thread_id: str | None = None
    #: Nodes executed, in order, by this call. A resumed run lists only what *it* ran,
    #: which is what makes "did not re-run" checkable by a caller and not just by a test.
    executed: list[str] = field(default_factory=list)


class _GraphState(BaseModel):
    """The bookkeeping one graph run persists alongside its user state.

    Serialized into the state message next to the user's own state. ``next_node`` is the
    load-bearing field: it is how resume knows where to pick up, and therefore which
    nodes must not run again.
    """

    #: The node to execute next. END means the run finished.
    next_node: str
    #: Nodes that have run to completion, in order.
    completed: list[str] = Field(default_factory=list)
    #: Nodes executed so far across all invocations, for the max_iterations budget.
    iterations: int = 0
    #: Serialized user state.
    state_json: str = ""


class StateGraph[StateT: BaseModel]:
    """A typed state graph, built node by node and then compiled.

    ``state_type`` is the pydantic model that flows through the graph; every node and
    router is typed against it. Nothing runs until :meth:`compile`, which is where the
    structural checks happen — see :meth:`compile` for exactly what it rejects.

    Example::

        graph = StateGraph(State)
        graph.add_node("fetch", fetch)
        graph.add_node("summarize", summarize)
        graph.set_entry_point("fetch")
        graph.add_edge("fetch", "summarize")
        graph.add_edge("summarize", END)
        compiled = graph.compile()

    ``add_node`` / ``add_edge`` / ``add_conditional_edges`` / ``set_entry_point`` all
    return ``self``, so the same graph can be written as a chain if you prefer.
    """

    def __init__(self, state_type: type[StateT]) -> None:
        if not (isinstance(state_type, type) and issubclass(state_type, BaseModel)):
            raise TypeError(
                f"StateGraph needs a pydantic model class as its state type, got "
                f"{state_type!r}. Define one:\n"
                "    class State(BaseModel):\n"
                "        question: str\n"
                "    graph = StateGraph(State)"
            )
        validate_append_fields(state_type)
        self.state_type = state_type
        self._nodes: dict[str, _Node[StateT]] = {}
        self._entry_point: str | None = None
        self._append = append_fields(state_type)

    def add_node(self, name: str, fn: NodeFn[StateT]) -> StateGraph[StateT]:
        """Register ``fn`` under ``name``.

        ``fn`` must be an async callable taking the graph's state and returning a partial
        update, a whole state, or None. An :class:`~actants.agents.agent.Agent` is usable
        here by wrapping it — see :func:`~actants.graph.agent_node.agent_node`.
        """
        if name == END:
            raise GraphValidationError(
                f"{END!r} is the reserved terminal node name and cannot be used for a "
                "node. Point an edge at END to finish the run instead."
            )
        if not name or not isinstance(name, str):
            raise GraphValidationError(f"Node names must be non-empty strings, got {name!r}.")
        if name in self._nodes:
            raise GraphValidationError(
                f"A node named {name!r} is already registered. Node names must be "
                "unique; pick another name, or remove the earlier add_node call."
            )
        if not callable(fn):
            raise GraphValidationError(
                f"The function for node {name!r} is not callable (got {type(fn).__name__!r}). "
                "A node is an async function taking the state and returning an update."
            )
        self._nodes[name] = _Node(name=name, fn=fn)
        return self

    def set_entry_point(self, name: str) -> StateGraph[StateT]:
        """Choose the node the run starts at."""
        self._entry_point = name
        return self

    def add_edge(self, start: str, end: str) -> StateGraph[StateT]:
        """Always go from ``start`` to ``end`` after ``start`` completes.

        ``end`` may be :data:`~actants.graph.state.END` to finish the run.
        """
        node = self._nodes.get(start)
        if node is None:
            raise GraphValidationError(
                f"add_edge({start!r}, {end!r}): {start!r} is not a registered node. "
                f"Registered nodes: {sorted(self._nodes) or ['<none>']}. "
                "Call add_node before adding edges from it."
            )
        if node.router is not None:
            raise GraphValidationError(
                f"Node {start!r} already has conditional edges, so it cannot also have "
                f"an unconditional edge to {end!r}. A node routes one way or the other: "
                "either add_edge, or add_conditional_edges with END in its mapping."
            )
        if node.static_target is not None:
            raise GraphValidationError(
                f"Node {start!r} already has an edge to {node.static_target!r}. "
                "A node has at most one unconditional edge; use add_conditional_edges "
                "to pick between several targets."
            )
        node.static_target = end
        return self

    def add_conditional_edges(
        self,
        start: str,
        router: RouterFn[StateT],
        mapping: Mapping[str, str],
    ) -> StateGraph[StateT]:
        """Route out of ``start`` by what ``router(state)`` returns.

        ``router`` returns a key of ``mapping``; the mapped value is the next node, or
        :data:`~actants.graph.state.END`. The indirection through ``mapping`` is what
        makes the branch inspectable: ``compile`` can check every reachable target
        exists, which it could not do if the router returned node names directly.
        """
        node = self._nodes.get(start)
        if node is None:
            raise GraphValidationError(
                f"add_conditional_edges({start!r}, ...): {start!r} is not a registered "
                f"node. Registered nodes: {sorted(self._nodes) or ['<none>']}. "
                "Call add_node before adding edges from it."
            )
        if node.static_target is not None:
            raise GraphValidationError(
                f"Node {start!r} already has an unconditional edge to "
                f"{node.static_target!r}, so it cannot also have conditional edges. "
                "Remove the add_edge call and put that target in the mapping."
            )
        if node.router is not None:
            raise GraphValidationError(
                f"Node {start!r} already has conditional edges. Pass every branch in a "
                "single mapping rather than calling add_conditional_edges twice."
            )
        if not callable(router):
            raise GraphValidationError(
                f"The router for node {start!r} is not callable (got "
                f"{type(router).__name__!r}). A router takes the state and returns a "
                "key of the mapping."
            )
        if not mapping:
            raise GraphValidationError(
                f"add_conditional_edges({start!r}, ...) was given an empty mapping, so "
                "no branch could ever be taken. Map at least one router result to a "
                f"node or to END, e.g. {{'done': END, 'again': {start!r}}}."
            )
        node.router = router
        node.routes = dict(mapping)
        return self

    def compile(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        interrupt_before: Iterable[str] | None = None,
        max_iterations: int = 25,
    ) -> CompiledGraph[StateT]:
        """Validate the graph's shape and return a runnable form.

        Rejects, before anything can run:

        - no entry point, or an entry point that is not a registered node
        - an edge or conditional target naming a node that does not exist
        - a node with no outgoing edge (it would silently end the run)
        - a node unreachable from the entry point
        - an ``interrupt_before`` naming a node that does not exist

        ``max_iterations`` caps how many nodes one run may execute. Graphs loop by
        design, so there is no structural way to tell a slow convergence from an
        infinite one; exceeding the cap raises
        :class:`~actants.errors.GraphRecursionError` naming the node that was running.
        """
        if not self._nodes:
            raise GraphValidationError(
                "This graph has no nodes. Add at least one with "
                "graph.add_node('name', fn) before compiling."
            )
        if self._entry_point is None:
            raise GraphValidationError(
                "This graph has no entry point, so there is nothing to run first. "
                f"Call graph.set_entry_point(name) with one of: {sorted(self._nodes)}."
            )
        if self._entry_point not in self._nodes:
            raise GraphValidationError(
                f"The entry point {self._entry_point!r} is not a registered node. "
                f"Registered nodes: {sorted(self._nodes)}."
            )
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise GraphValidationError(
                f"max_iterations must be an integer, got {max_iterations!r}."
            )
        if max_iterations < 1:
            raise GraphValidationError(
                f"max_iterations must be >= 1, got {max_iterations}. It caps how many "
                "nodes one run may execute before actants assumes the graph is stuck."
            )

        self._validate_targets()
        self._validate_reachability()

        guarded = frozenset(interrupt_before or ())
        if isinstance(interrupt_before, str):
            raise GraphValidationError(
                "interrupt_before must be a collection of node names, not a single "
                f"string ({interrupt_before!r} would be read one character at a time). "
                f"Use interrupt_before=[{interrupt_before!r}]."
            )
        unknown = sorted(guarded - set(self._nodes))
        if unknown:
            raise GraphValidationError(
                f"interrupt_before names node(s) {unknown} that this graph does not "
                f"have. Registered nodes: {sorted(self._nodes)}."
            )
        if checkpointer is not None and not isinstance(checkpointer, Checkpointer):
            raise TypeError(
                f"checkpointer must implement the Checkpointer protocol (put/get/"
                f"list_threads/delete), got {type(checkpointer).__name__!r}. "
                "Example: graph.compile(checkpointer=SqliteCheckpointer('runs.db'))."
            )

        return CompiledGraph(
            state_type=self.state_type,
            nodes=dict(self._nodes),
            entry_point=self._entry_point,
            append=self._append,
            checkpointer=checkpointer,
            interrupt_before=guarded,
            max_iterations=max_iterations,
        )

    def _validate_targets(self) -> None:
        """Every edge target must be a real node or END, and every node must have one."""
        for node in self._nodes.values():
            if node.static_target is None and node.router is None:
                raise GraphValidationError(
                    f"Node {node.name!r} has no outgoing edge, so the run would stop "
                    "there without reaching END. Add graph.add_edge("
                    f"{node.name!r}, END) if it is a terminal node, or an edge to "
                    "whatever runs next."
                )
            if node.static_target is not None and not self._is_target(node.static_target):
                raise GraphValidationError(
                    f"Node {node.name!r} has an edge to {node.static_target!r}, which is "
                    f"not a registered node. Registered nodes: {sorted(self._nodes)}, "
                    f"plus {END!r} to finish the run."
                )
            for key, target in sorted(node.routes.items()):
                if not self._is_target(target):
                    raise GraphValidationError(
                        f"Node {node.name!r} routes {key!r} to {target!r}, which is not "
                        f"a registered node. Registered nodes: {sorted(self._nodes)}, "
                        f"plus {END!r} to finish the run."
                    )

    def _is_target(self, name: str) -> bool:
        return name == END or name in self._nodes

    def _validate_reachability(self) -> None:
        """Reject nodes the entry point cannot reach.

        An unreachable node is dead code that reads as live code — nearly always a typo
        in an edge or a branch someone forgot to wire up.
        """
        assert self._entry_point is not None
        seen: set[str] = set()
        frontier = [self._entry_point]
        while frontier:
            current = frontier.pop()
            if current in seen or current == END:
                continue
            seen.add(current)
            node = self._nodes[current]
            if node.static_target is not None:
                frontier.append(node.static_target)
            frontier.extend(node.routes.values())

        unreachable = sorted(set(self._nodes) - seen)
        if unreachable:
            raise GraphValidationError(
                f"Node(s) {unreachable} cannot be reached from the entry point "
                f"{self._entry_point!r}, so they would never run. Add an edge to them, "
                "or remove them."
            )


class CompiledGraph[StateT: BaseModel]:
    """A validated graph, ready to run. Built by :meth:`StateGraph.compile`.

    Runs are driven by :meth:`invoke` (await the final state), :meth:`stream` (watch
    node-by-node), and :meth:`resume` (continue a checkpointed run).
    """

    def __init__(
        self,
        *,
        state_type: type[StateT],
        nodes: dict[str, _Node[StateT]],
        entry_point: str,
        append: frozenset[str],
        checkpointer: Checkpointer | None,
        interrupt_before: frozenset[str],
        max_iterations: int,
    ) -> None:
        self.state_type = state_type
        self.entry_point = entry_point
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.max_iterations = max_iterations
        self._nodes = nodes
        self._append = append

    @property
    def nodes(self) -> frozenset[str]:
        """The names of every node in this graph. **Read-only.**"""
        return frozenset(self._nodes)

    async def invoke(
        self,
        state: StateT,
        *,
        thread_id: str | None = None,
    ) -> GraphResult[StateT]:
        """Run the graph from its entry point until END, or until a pause.

        Pass ``thread_id`` — with a ``checkpointer`` on the compiled graph — to make the
        run durable. State is persisted after every node completes, so :meth:`resume`
        can continue it without re-running the nodes that already finished. A
        ``thread_id`` without a checkpointer is a ``ValueError``; a checkpointer without
        a ``thread_id`` persists nothing, so durability stays opt-in per run.
        """
        run = self._new_run(state, thread_id)
        async for _ in self._drive(run):
            pass
        return self._result(run)

    async def stream(
        self,
        state: StateT,
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[GraphEvent[StateT]]:
        """Run the graph, yielding an event as each node starts and finishes.

        Yields, in order:
          - ``GraphNodeStarted(node, iteration)`` — before a node runs
          - ``GraphNodeCompleted(node, iteration, state)`` — after its update merged
          - ``GraphInterrupted(node, state)`` — paused before a guarded node (terminal)
          - ``GraphCompleted(state)`` — reached END (terminal)

        Every run ends in exactly one of the two terminal events.
        """
        run = self._new_run(state, thread_id)
        async for event in self._drive(run):
            yield event

    def _new_run(self, state: StateT, thread_id: str | None) -> _Run[StateT]:
        """Validate the entry conditions of a fresh run and build its bookkeeping."""
        if thread_id is not None and self.checkpointer is None:
            raise ValueError(
                f"invoke(thread_id={thread_id!r}) needs a checkpointer to persist to, "
                "and this graph was compiled without one. Compile it with "
                "graph.compile(checkpointer=SqliteCheckpointer('runs.db')), or drop "
                "thread_id for a non-durable run."
            )
        if not isinstance(state, self.state_type):
            raise TypeError(
                f"invoke() expects a {self.state_type.__name__} instance, got "
                f"{type(state).__name__!r}. The graph was built with "
                f"StateGraph({self.state_type.__name__})."
            )
        return _Run(
            state=state,
            next_node=self.entry_point,
            completed=[],
            iterations=0,
            thread_id=thread_id,
        )

    def _result(self, run: _Run[StateT]) -> GraphResult[StateT]:
        """Render a finished or paused run as the caller-facing result."""
        return GraphResult(
            state=run.state,
            interrupted=run.pending_node is not None,
            pending_node=run.pending_node,
            thread_id=run.thread_id,
            executed=list(run.executed),
        )

    async def resume(
        self,
        thread_id: str,
        *,
        approve: bool | None = None,
    ) -> GraphResult[StateT]:
        """Continue a checkpointed run from the last node that completed.

        **The guarantee.** Every node whose completion was recorded before the crash is
        skipped, never re-run — resume is at-most-once for them. The run picks up at the
        node the checkpoint says was next, with the state exactly as the last completed
        node left it.

        ``approve`` answers a run paused by ``interrupt_before``: ``True`` runs the
        pending node and continues, ``False`` skips it and routes onward as though it had
        run without changing the state. Mirrors
        :meth:`~actants.agents.agent.Agent.resume`.

        Resuming a thread that already completed returns its stored state without
        re-running anything. An unknown ``thread_id`` raises
        :class:`~actants.errors.UnknownThreadError`.

        Two processes resuming the same ``thread_id`` concurrently is undefined; actants
        does not lock a thread across processes.
        """
        resumption = await self._resume_run(thread_id, approve)
        if resumption.finished:
            return self._result(resumption.run)
        async for _ in self._drive(
            resumption.run,
            skip_pending=resumption.skip_pending,
            approved=resumption.approved,
        ):
            pass
        return self._result(resumption.run)

    async def resume_stream(
        self,
        thread_id: str,
        *,
        approve: bool | None = None,
    ) -> AsyncIterator[GraphEvent[StateT]]:
        """Streaming form of :meth:`resume`; yields the same events as :meth:`stream`."""
        resumption = await self._resume_run(thread_id, approve)
        if resumption.finished:
            yield GraphCompleted(state=resumption.run.state)
            return
        async for event in self._drive(
            resumption.run,
            skip_pending=resumption.skip_pending,
            approved=resumption.approved,
        ):
            yield event

    async def _resume_run(self, thread_id: str, approve: bool | None) -> _Resumption[StateT]:
        """Load a checkpointed run and decide how it continues."""
        if self.checkpointer is None:
            raise ValueError(
                f"resume({thread_id!r}) needs a checkpointer to read from, and this "
                "graph was compiled without one. Resume on a graph compiled with the "
                "same checkpointer the run was started with."
            )

        checkpoint = await self.checkpointer.get(thread_id)
        if checkpoint is None:
            known = await self.checkpointer.list_threads()
            listed = ", ".join(sorted(known)[:10]) or "<none>"
            raise UnknownThreadError(
                f"No checkpoint for thread_id {thread_id!r}. Either it never ran under "
                f"this checkpointer, or its state was deleted. Known threads: {listed}."
            )

        stored = self._read(checkpoint, thread_id)
        state = state_from_json(self.state_type, stored.state_json)
        run = _Run(
            state=state,
            next_node=stored.next_node,
            completed=list(stored.completed),
            iterations=stored.iterations,
            thread_id=thread_id,
        )

        if checkpoint.status == "completed":
            return _Resumption(run=run, skip_pending=False, approved=False, finished=True)
        if checkpoint.status == "failed":
            raise RuntimeError(
                f"Thread {thread_id!r} is checkpointed as failed and cannot be resumed: "
                f"{checkpoint.error}. Start a new run, or delete the thread first."
            )
        if checkpoint.status == "interrupted" and approve is None:
            raise ValueError(
                f"Thread {thread_id!r} is paused before node {stored.next_node!r} and "
                f"needs a decision. Call resume({thread_id!r}, approve=True) to run it, "
                "or approve=False to skip it and continue past it."
            )

        # A rejected node is treated as having run and changed nothing, so the run
        # continues past it rather than dying — the same shape as the agent recording a
        # rejected tool call and letting the model react to the refusal.
        paused = checkpoint.status == "interrupted"
        return _Resumption(
            run=run,
            skip_pending=paused and approve is False,
            approved=paused and approve is True,
            finished=False,
        )

    async def _drive(
        self,
        run: _Run[StateT],
        *,
        skip_pending: bool = False,
        approved: bool = False,
    ) -> AsyncIterator[GraphEvent[StateT]]:
        """The node loop shared by every entry point.

        ``skip_pending`` routes past ``run.next_node`` without running it, which is what
        a rejected interrupt means. ``approved`` lets the *first* node through the
        interrupt check — without it, resuming with approve=True would pause again on
        the very node the approval was for, and the run could never move past it.
        """
        run.pending_node = None
        try:
            while run.next_node != END:
                node = self._nodes[run.next_node]

                if skip_pending:
                    skip_pending = False
                    run.completed.append(node.name)
                    run.next_node = self._next_of(node, run.state)
                    await self._checkpoint(run, status="running")
                    continue

                if run.iterations >= self.max_iterations:
                    raise GraphRecursionError(
                        f"Graph exceeded max_iterations={self.max_iterations} without "
                        f"reaching END; it was about to run node {node.name!r} again. "
                        "A loop is not converging — check the router that keeps "
                        f"selecting {node.name!r}, or raise the cap with "
                        "graph.compile(max_iterations=...).",
                        node=node.name,
                        iterations=run.iterations,
                    )

                if node.name in self.interrupt_before and not approved:
                    yield await self._interrupt(run, node.name)
                    return
                # Consumed by the node it was granted for; a later visit to the same
                # guarded node — a loop coming back around — pauses again as it should.
                approved = False

                yield GraphNodeStarted(node=node.name, iteration=run.iterations)
                run.state = await self._run_node(node, run.state)
                run.iterations += 1
                run.completed.append(node.name)
                run.executed.append(node.name)
                run.next_node = self._next_of(node, run.state)
                # After the node's update has landed, so the durable record says this
                # node is done and resume must not run it again.
                await self._checkpoint(run, status="running")
                yield GraphNodeCompleted(
                    node=node.name, iteration=run.iterations - 1, state=run.state
                )
        except Exception as exc:
            await self._checkpoint_failure(run, exc)
            raise

        await self._checkpoint(run, status="completed")
        yield GraphCompleted(state=run.state)

    async def _run_node(self, node: _Node[StateT], state: StateT) -> StateT:
        """Run one node and merge whatever it returned into the state."""
        update = await node.fn(state)
        if update is None:
            return state
        if isinstance(update, self.state_type):
            return update
        if not isinstance(update, dict):
            raise GraphValidationError(
                f"Node {node.name!r} returned {type(update).__name__!r}. A node must "
                f"return a dict of state updates, a {self.state_type.__name__} "
                "instance, or None."
            )
        return merge_update(
            state,
            update,
            accumulate=self._append,
            node=node.name,
        )

    def _next_of(self, node: _Node[StateT], state: StateT) -> str:
        """Decide what runs after ``node``, following its edge or asking its router."""
        if node.router is not None:
            key = node.router(state)
            if key not in node.routes:
                raise GraphValidationError(
                    f"The router for node {node.name!r} returned {key!r}, which is not "
                    f"a key of its mapping. Known keys: {sorted(node.routes)}. "
                    "A router must return one of the mapping's keys."
                )
            return node.routes[key]
        assert node.static_target is not None  # compile() rejects a node with neither
        return node.static_target

    async def _interrupt(self, run: _Run[StateT], node: str) -> GraphInterrupted[StateT]:
        """Persist the paused run and report the node it stopped in front of."""
        if self.checkpointer is None or run.thread_id is None:
            raise ValueError(
                f"Node {node!r} is in interrupt_before, but this run has no checkpointer "
                "and thread_id to persist the pause to — there would be nothing to "
                "resume from. Compile the graph with a checkpointer and call "
                "invoke(state, thread_id=...)."
            )
        run.pending_node = node
        await self._checkpoint(run, status="interrupted")
        return GraphInterrupted(node=node, state=run.state)

    async def _checkpoint(self, run: _Run[StateT], *, status: CheckpointStatus) -> None:
        """Persist the run's state, or do nothing if this run is not durable."""
        if self.checkpointer is None or run.thread_id is None:
            return
        payload = _GraphState(
            next_node=run.next_node,
            completed=list(run.completed),
            iterations=run.iterations,
            state_json=state_to_json(run.state),
        )
        await self.checkpointer.put(
            Checkpoint(
                thread_id=run.thread_id,
                status=status,
                messages=[ChatMessage(role=_STATE_ROLE, content=payload.model_dump_json())],
                tag=GRAPH_TAG,
                created_at=run.created_at,
            )
        )

    async def _checkpoint_failure(self, run: _Run[StateT], exc: BaseException) -> None:
        """Mark the thread failed, preserving what was already durably recorded.

        Flips the stored checkpoint's status rather than writing fresh state: the last
        good checkpoint is the one worth keeping, and a store that itself fails here is
        swallowed so the original exception is what the caller sees.
        """
        if self.checkpointer is None or run.thread_id is None:
            return
        with contextlib.suppress(Exception):
            stored = await self.checkpointer.get(run.thread_id)
            if stored is None:
                return
            stored.status = "failed"
            stored.error = f"{type(exc).__name__}: {exc}"
            stored.updated_at = time.time()
            await self.checkpointer.put(stored)

    def _read(self, checkpoint: Checkpoint, thread_id: str) -> _GraphState:
        """Pull graph bookkeeping out of a checkpoint, refusing an agent's."""
        if checkpoint.tag != GRAPH_TAG or not checkpoint.messages:
            raise GraphValidationError(
                f"Thread {thread_id!r} was not written by a graph run — it looks like an "
                "Agent checkpoint. Resume it with Agent.resume, or use a different "
                "thread_id for this graph."
            )
        return _GraphState.model_validate_json(checkpoint.messages[0].content)


@dataclass
class _Resumption[StateT: BaseModel]:
    """How a loaded checkpoint continues: the run, and what to do with its pending node.

    ``finished`` short-circuits everything else — the thread already reached END, so its
    stored state is handed back without running a single node.
    """

    run: _Run[StateT]
    skip_pending: bool
    approved: bool
    finished: bool


@dataclass
class _Run[StateT: BaseModel]:
    """The mutable bookkeeping one graph run carries.

    ``thread_id`` is None for a non-durable run, and every checkpoint write is a no-op in
    that case — which keeps the un-threaded path identical to a graph with no
    checkpointer at all.

    ``executed`` lives here rather than on the CompiledGraph because one compiled graph
    may be driving several concurrent runs, and a per-graph field would report another
    run's nodes.
    """

    state: StateT
    next_node: str
    completed: list[str]
    iterations: int
    thread_id: str | None
    executed: list[str] = field(default_factory=list)
    #: Set when the run stopped in front of a guarded node; None otherwise.
    pending_node: str | None = None
    created_at: float = field(default_factory=time.time)


__all__ = [
    "CompiledGraph",
    "GraphResult",
    "NodeFn",
    "RouterFn",
    "StateGraph",
]
