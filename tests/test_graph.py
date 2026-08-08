"""Typed state graphs: control flow, compile-time validation, and durability.

The guarantee under test mirrors the agent's: resume is at-most-once for every node that
completed before the crash. Most durability tests here assert against a per-node call
counter rather than the final state, because "did not re-run" is the whole product and a
state assertion would not catch a node whose side effect happened twice.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from actants.agents.agent import Agent
from actants.agents.checkpoint import (
    RESUME_FAILED_ACKNOWLEDGED,
    Checkpoint,
    InMemoryCheckpointer,
    SqliteCheckpointer,
)
from actants.errors import (
    ActantsError,
    GraphError,
    GraphRecursionError,
    GraphValidationError,
    UnknownThreadError,
)
from actants.graph import END, Append, CompiledGraph, StateGraph, agent_node
from actants.graph.state import merge_update
from actants.llm.client import LLM
from actants.testing import FakeLLMProvider, fake_completion


class State(BaseModel):
    """The state model most tests here run over."""

    value: int = 0
    label: str = ""
    log: Annotated[list[str], Append] = Field(default_factory=list)


class Counter:
    """Hands out node functions that record every execution.

    ``count`` is the assertion that matters in the durability tests: a node that ran
    twice shows up here and nowhere else.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def node(self, name: str, **update: Any) -> Any:
        async def run(state: State) -> dict[str, Any]:
            self.names.append(name)
            return dict(update)

        return run

    def count(self, name: str) -> int:
        return self.names.count(name)


def _linear(counter: Counter) -> StateGraph[State]:
    """a -> b -> END, the smallest useful graph."""
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", value=1))
    graph.add_node("b", counter.node("b", label="done"))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph


# ---------------------------------------------------------------------------
# Linear flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_linear_graph_runs_every_node_in_order() -> None:
    counter = Counter()
    compiled = _linear(counter).compile()

    result = await compiled.invoke(State())

    assert result.state.value == 1
    assert result.state.label == "done"
    assert result.executed == ["a", "b"]
    assert result.interrupted is False
    assert result.pending_node is None
    assert result.thread_id is None


@pytest.mark.asyncio
async def test_node_returning_none_leaves_state_untouched() -> None:
    async def noop(state: State) -> None:
        return None

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("noop", noop)
    graph.set_entry_point("noop")
    graph.add_edge("noop", END)

    result = await graph.compile().invoke(State(value=7))
    assert result.state.value == 7


@pytest.mark.asyncio
async def test_node_may_return_a_whole_state() -> None:
    async def rebuild(state: State) -> State:
        return State(value=99, label="rebuilt")

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("rebuild", rebuild)
    graph.set_entry_point("rebuild")
    graph.add_edge("rebuild", END)

    result = await graph.compile().invoke(State())
    assert result.state.value == 99
    assert result.state.label == "rebuilt"


@pytest.mark.asyncio
async def test_the_initial_state_object_is_not_mutated() -> None:
    """Nodes merge into a copy, so the caller's object is still usable afterwards."""
    counter = Counter()
    compiled = _linear(counter).compile()
    initial = State(value=0)

    await compiled.invoke(initial)

    assert initial.value == 0, "invoke() mutated the caller's state object"


@pytest.mark.asyncio
async def test_a_node_returning_an_unknown_field_is_an_error() -> None:
    """A dict's keys escape the type checker, so this is the only place a typo is caught."""

    async def typo(state: State) -> dict[str, Any]:
        return {"valeu": 1}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("typo", typo)
    graph.set_entry_point("typo")
    graph.add_edge("typo", END)

    with pytest.raises(GraphValidationError) as exc:
        await graph.compile().invoke(State())
    message = str(exc.value)
    assert "valeu" in message
    assert "'value'" in message, "the error should list the fields that do exist"


@pytest.mark.asyncio
async def test_invoke_rejects_the_wrong_state_type() -> None:
    class Other(BaseModel):
        q: str = ""

    compiled = _linear(Counter()).compile()
    with pytest.raises(TypeError, match="State"):
        await compiled.invoke(Other())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_reducer_replaces_a_field() -> None:
    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", value=1))
    graph.add_node("b", counter.node("b", value=2))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = await graph.compile().invoke(State())
    assert result.state.value == 2


@pytest.mark.asyncio
async def test_append_reducer_accumulates_across_nodes() -> None:
    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", log=["one"]))
    graph.add_node("b", counter.node("b", log=["two"]))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = await graph.compile().invoke(State(log=["zero"]))
    assert result.state.log == ["zero", "one", "two"]


@pytest.mark.asyncio
async def test_append_reducer_takes_a_bare_value_as_one_item() -> None:
    """A string must become one entry, not be spread into characters."""
    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", log="solo"))
    graph.set_entry_point("a")
    graph.add_edge("a", END)

    result = await graph.compile().invoke(State())
    assert result.state.log == ["solo"]


@pytest.mark.asyncio
async def test_concurrent_runs_from_one_seed_state_do_not_corrupt_each_other() -> None:
    """Regression: ``model_copy(update=...)`` is shallow, so runs shared the seed's lists.

    Two runs seeded from one state object each appended one item in place and both saw
    two — the consequence that matters, and one no identity assertion would have caught.
    """

    async def mutate(state: State) -> dict[str, Any]:
        await asyncio.sleep(0)
        state.log.append("mine")
        return {}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("m", mutate)
    graph.set_entry_point("m")
    graph.add_edge("m", END)
    compiled = graph.compile()

    seed = State()
    first, second = await asyncio.gather(compiled.invoke(seed), compiled.invoke(seed))

    assert first.state.log == ["mine"], "a concurrent run's write leaked into this one"
    assert second.state.log == ["mine"]
    assert seed.log == [], "invoke wrote through the caller's seed object"


def test_merge_update_does_not_share_containers_with_the_state_it_copied() -> None:
    """The documented 'never mutates state' contract, asserted on a real write."""
    before = State(log=["seed"])
    after = merge_update(before, {"value": 1}, accumulate=frozenset(), node="n")

    after.log.append("added")
    assert before.log == ["seed"], "the pre-node state saw a write made after it"


def test_append_on_a_non_list_field_is_rejected() -> None:
    class Bad(BaseModel):
        n: Annotated[int, Append] = 0

    with pytest.raises(GraphValidationError) as exc:
        StateGraph(Bad)
    assert "not a list" in str(exc.value)


# ---------------------------------------------------------------------------
# Conditional branching and loops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "expected"), [(1, "high"), (-1, "low")])
async def test_conditional_edges_pick_a_branch(value: int, expected: str) -> None:
    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("start", counter.node("start"))
    graph.add_node("high", counter.node("high", label="high"))
    graph.add_node("low", counter.node("low", label="low"))
    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start",
        lambda s: "high" if s.value > 0 else "low",
        {"high": "high", "low": "low"},
    )
    graph.add_edge("high", END)
    graph.add_edge("low", END)

    result = await graph.compile().invoke(State(value=value))

    assert result.state.label == expected
    assert result.executed == ["start", expected]
    assert counter.count("high" if expected == "low" else "low") == 0


@pytest.mark.asyncio
async def test_a_loop_terminates_when_its_router_says_so() -> None:
    async def increment(state: State) -> dict[str, Any]:
        return {"value": state.value + 1, "log": [f"pass-{state.value + 1}"]}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("work", increment)
    graph.set_entry_point("work")
    graph.add_conditional_edges(
        "work",
        lambda s: "done" if s.value >= 3 else "again",
        {"again": "work", "done": END},
    )

    result = await graph.compile().invoke(State())

    assert result.state.value == 3
    assert result.state.log == ["pass-1", "pass-2", "pass-3"]
    assert result.executed == ["work", "work", "work"]


@pytest.mark.asyncio
async def test_a_router_returning_an_unmapped_key_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.set_entry_point("a")
    graph.add_conditional_edges("a", lambda s: "surprise", {"done": END})

    with pytest.raises(GraphValidationError) as exc:
        await graph.compile().invoke(State())
    message = str(exc.value)
    assert "surprise" in message
    assert "done" in message, "the error should list the keys that are mapped"


# ---------------------------------------------------------------------------
# Cycle safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_trips_and_names_the_node_that_spun() -> None:
    """A router that never returns END must fail loudly, naming the culprit."""
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("spin", Counter().node("spin"))
    graph.set_entry_point("spin")
    graph.add_conditional_edges("spin", lambda s: "again", {"again": "spin"})

    with pytest.raises(GraphRecursionError) as exc:
        await graph.compile(max_iterations=5).invoke(State())

    assert exc.value.node == "spin"
    assert exc.value.iterations == 5
    message = str(exc.value)
    assert "max_iterations=5" in message
    assert "'spin'" in message


@pytest.mark.asyncio
async def test_max_iterations_does_not_fire_on_a_graph_that_converges() -> None:
    """The cap must not punish a loop that finishes just inside its budget."""

    async def increment(state: State) -> dict[str, Any]:
        return {"value": state.value + 1}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("work", increment)
    graph.set_entry_point("work")
    graph.add_conditional_edges(
        "work",
        lambda s: "done" if s.value >= 3 else "again",
        {"again": "work", "done": END},
    )

    result = await graph.compile(max_iterations=3).invoke(State())
    assert result.state.value == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_max_iterations_must_be_positive(bad: int) -> None:
    with pytest.raises(GraphValidationError, match="max_iterations"):
        _linear(Counter()).compile(max_iterations=bad)


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_compiling_a_graph_with_no_nodes_is_an_error() -> None:
    with pytest.raises(GraphValidationError, match="no nodes"):
        StateGraph(State).compile()


def test_compiling_without_an_entry_point_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.add_edge("a", END)

    with pytest.raises(GraphValidationError) as exc:
        graph.compile()
    assert "entry point" in str(exc.value)
    assert "'a'" in str(exc.value), "the error should list the candidates"


def test_an_entry_point_that_is_not_a_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.add_edge("a", END)
    graph.set_entry_point("nope")

    with pytest.raises(GraphValidationError, match="entry point 'nope'"):
        graph.compile()


def test_an_edge_to_an_undefined_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.set_entry_point("a")
    graph.add_edge("a", "ghost")

    with pytest.raises(GraphValidationError) as exc:
        graph.compile()
    assert "'ghost'" in str(exc.value)


def test_a_conditional_mapping_naming_an_undefined_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.set_entry_point("a")
    graph.add_conditional_edges("a", lambda s: "x", {"x": "ghost", "done": END})

    with pytest.raises(GraphValidationError) as exc:
        graph.compile()
    assert "'ghost'" in str(exc.value)


def test_an_unreachable_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.add_node("orphan", Counter().node("orphan"))
    graph.set_entry_point("a")
    graph.add_edge("a", END)
    graph.add_edge("orphan", END)

    with pytest.raises(GraphValidationError) as exc:
        graph.compile()
    assert "orphan" in str(exc.value)
    assert "cannot be reached" in str(exc.value)


def test_a_node_with_no_outgoing_edge_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.set_entry_point("a")

    with pytest.raises(GraphValidationError) as exc:
        graph.compile()
    assert "no outgoing edge" in str(exc.value)


def test_adding_an_edge_from_an_undefined_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    with pytest.raises(GraphValidationError, match="not a registered node"):
        graph.add_edge("ghost", END)


def test_adding_conditional_edges_from_an_undefined_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    with pytest.raises(GraphValidationError, match="not a registered node"):
        graph.add_conditional_edges("ghost", lambda s: "x", {"x": END})


def test_a_duplicate_node_name_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    counter = Counter()
    graph.add_node("a", counter.node("a"))
    with pytest.raises(GraphValidationError, match="already registered"):
        graph.add_node("a", counter.node("a"))


def test_a_node_may_not_be_named_end() -> None:
    graph: StateGraph[State] = StateGraph(State)
    with pytest.raises(GraphValidationError, match="reserved"):
        graph.add_node(END, Counter().node("x"))


def test_a_node_may_not_shadow_the_routing_marker() -> None:
    """It is a checkpoint value, so a node of that name would be dispatched on resume."""
    graph: StateGraph[State] = StateGraph(State)
    with pytest.raises(GraphValidationError, match="reserved"):
        graph.add_node("__routing__", Counter().node("x"))


def test_mixing_conditional_and_unconditional_edges_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.set_entry_point("a")
    graph.add_edge("a", END)

    with pytest.raises(GraphValidationError, match="already has an unconditional edge"):
        graph.add_conditional_edges("a", lambda s: "x", {"x": END})


def test_two_unconditional_edges_from_one_node_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    graph.add_node("b", Counter().node("b"))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")

    with pytest.raises(GraphValidationError, match="at most one unconditional edge"):
        graph.add_edge("a", END)


def test_an_empty_conditional_mapping_is_an_error() -> None:
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", Counter().node("a"))
    with pytest.raises(GraphValidationError, match="empty mapping"):
        graph.add_conditional_edges("a", lambda s: "x", {})


def test_interrupt_before_an_undefined_node_is_an_error() -> None:
    with pytest.raises(GraphValidationError, match="ghost"):
        _linear(Counter()).compile(interrupt_before=["ghost"])


def test_interrupt_before_rejects_a_bare_string() -> None:
    with pytest.raises(GraphValidationError, match="one character at a time"):
        _linear(Counter()).compile(interrupt_before="a")  # type: ignore[arg-type]


def test_state_graph_rejects_a_non_model_state_type() -> None:
    with pytest.raises(TypeError, match="pydantic model"):
        StateGraph(dict)  # type: ignore[type-var]


def test_compile_rejects_a_checkpointer_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="Checkpointer protocol"):
        _linear(Counter()).compile(checkpointer=object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_graph_without_a_thread_id_persists_nothing() -> None:
    """Durability is opt-in per run, not a mode the graph is in."""
    store = InMemoryCheckpointer()
    compiled = _linear(Counter()).compile(checkpointer=store)

    await compiled.invoke(State())

    assert await store.list_threads() == []


@pytest.mark.asyncio
async def test_a_thread_id_without_a_checkpointer_is_rejected() -> None:
    compiled = _linear(Counter()).compile()
    with pytest.raises(ValueError, match="checkpointer"):
        await compiled.invoke(State(), thread_id="t1")


@pytest.mark.asyncio
async def test_state_is_checkpointed_after_every_node() -> None:
    """The boundary the resume guarantee rests on: each node is durable once it is done.

    Asserted as "every completion is visible in a checkpoint before the next node runs",
    not as a write count — the number of writes per node is an implementation detail, and
    pinning it hid the router bug that lost a completed node's record entirely.
    """
    seen: list[list[str]] = []

    class Recording(InMemoryCheckpointer):
        async def put(self, checkpoint: Checkpoint) -> None:
            payload = checkpoint.messages[0].content
            seen.append(list(json.loads(payload)["completed"]))
            await super().put(checkpoint)

    compiled = _linear(Counter()).compile(checkpointer=Recording())
    await compiled.invoke(State(), thread_id="t1")

    assert seen[0] == ["a"]  # 'a' durable before 'b' can start
    assert seen[-1] == ["a", "b"]
    assert [c for c in seen if c and c[-1] == "b"], "'b' never recorded as completed"


@pytest.mark.asyncio
async def test_a_completed_run_is_checkpointed_as_completed() -> None:
    store = InMemoryCheckpointer()
    compiled = _linear(Counter()).compile(checkpointer=store)
    await compiled.invoke(State(), thread_id="t1")

    checkpoint = await store.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "completed"
    assert checkpoint.thread_id == "t1"


@pytest.mark.asyncio
async def test_a_failed_node_marks_the_thread_failed_and_re_raises() -> None:
    async def boom(state: State) -> dict[str, Any]:
        raise RuntimeError("node exploded")

    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", value=1))
    graph.add_node("boom", boom)
    graph.set_entry_point("a")
    graph.add_edge("a", "boom")
    graph.add_edge("boom", END)

    store = InMemoryCheckpointer()
    with pytest.raises(RuntimeError, match="node exploded"):
        await graph.compile(checkpointer=store).invoke(State(), thread_id="t1")

    checkpoint = await store.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "failed"
    assert "node exploded" in (checkpoint.error or "")


@pytest.mark.asyncio
async def test_resume_does_not_rerun_a_node_that_already_completed() -> None:
    """The headline guarantee, asserted on a side-effect counter."""
    counter = Counter()

    def build(store: Any, *, explode: bool) -> CompiledGraph[State]:
        async def b(state: State) -> dict[str, Any]:
            counter.names.append("b")
            if explode:
                raise RuntimeError("process died")
            return {"label": "done"}

        graph: StateGraph[State] = StateGraph(State)
        graph.add_node("a", counter.node("a", value=1))
        graph.add_node("b", b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        return graph.compile(checkpointer=store)

    store = InMemoryCheckpointer()
    with pytest.raises(RuntimeError, match="process died"):
        await build(store, explode=True).invoke(State(), thread_id="t1")
    assert counter.count("a") == 1

    stored = await store.get("t1")
    assert stored is not None
    stored.status = "running"
    stored.error = None
    await store.put(stored)

    result = await build(store, explode=False).resume("t1")

    assert result.state.label == "done"
    assert result.state.value == 1, "the completed node's update must survive the crash"
    assert counter.count("a") == 1, "a node that already ran was executed again"
    assert result.executed == ["b"], "resume must run only what was left"


@pytest.mark.asyncio
async def test_a_router_that_raises_does_not_lose_its_nodes_completion() -> None:
    """Regression: an unmapped router key used to erase a completed node's side effect.

    ``_next_of`` runs between the node and the checkpoint, so a router raising left the
    node — whose external side effect had already landed — unrecorded, and resume ran it
    a second time. Asserted on the side-effect count, which is the only thing that
    distinguishes "recorded" from "shipped the order twice".
    """
    shipped: list[str] = []

    async def ship(state: State) -> dict[str, Any]:
        shipped.append("order")
        return {"label": "shipped"}

    def build(store: Any, *, broken: bool) -> CompiledGraph[State]:
        graph: StateGraph[State] = StateGraph(State)
        graph.add_node("ship", ship)
        graph.set_entry_point("ship")
        graph.add_conditional_edges(
            "ship",
            (lambda s: "typo") if broken else (lambda s: "done"),
            {"done": END},
        )
        return graph.compile(checkpointer=store)

    store = InMemoryCheckpointer()
    with pytest.raises(GraphValidationError, match="not a key of its mapping"):
        await build(store, broken=True).invoke(State(), thread_id="t1")
    assert shipped == ["order"]

    stored = await store.get("t1")
    assert stored is not None
    assert "ship" in json.loads(stored.messages[0].content)["completed"], (
        "the node completed, so the durable record must say so even though routing failed"
    )

    # The operator deploys the fixed router and resumes.
    stored.status = "running"
    stored.error = None
    await store.put(stored)
    result = await build(store, broken=False).resume("t1")

    assert shipped == ["order"], "the completed node's side effect happened twice"
    assert result.executed == [], "resume must not re-run a node that already completed"


@pytest.mark.asyncio
async def test_resume_across_a_fresh_sqlite_checkpointer(tmp_path: Path) -> None:
    """Proves the state lives in the file, not in the checkpointer object's memory.

    A second ``SqliteCheckpointer`` on the same path, driving a graph compiled fresh, is
    what a resume in a new process actually looks like.
    """
    db = tmp_path / "runs.db"
    counter = Counter()

    def build(store: Any, *, explode: bool) -> CompiledGraph[State]:
        async def b(state: State) -> dict[str, Any]:
            counter.names.append("b")
            if explode:
                raise RuntimeError("process died")
            return {"label": "done"}

        graph: StateGraph[State] = StateGraph(State)
        graph.add_node("a", counter.node("a", value=5))
        graph.add_node("b", b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        return graph.compile(checkpointer=store)

    with pytest.raises(RuntimeError, match="process died"):
        await build(SqliteCheckpointer(db), explode=True).invoke(State(), thread_id="job-42")

    reopened = SqliteCheckpointer(db)
    stored = await reopened.get("job-42")
    assert stored is not None
    assert stored.status == "failed"
    stored.status = "running"
    stored.error = None
    await reopened.put(stored)

    result = await build(SqliteCheckpointer(db), explode=False).resume("job-42")

    assert result.state.label == "done"
    assert result.state.value == 5, "state written before the crash must come back"
    assert counter.count("a") == 1, "the node that already ran must not run in the new process"
    assert result.executed == ["b"]


@pytest.mark.asyncio
async def test_resume_of_a_loop_keeps_its_accumulated_state(tmp_path: Path) -> None:
    """An Append field must come back with everything the crashed run put in it."""
    db = tmp_path / "runs.db"
    attempts = {"n": 0}

    def build() -> CompiledGraph[State]:
        async def work(state: State) -> dict[str, Any]:
            attempts["n"] += 1
            # Dies on the third visit of the first run only.
            if attempts["n"] == 3:
                raise RuntimeError("process died")
            return {"value": state.value + 1, "log": [f"pass-{state.value + 1}"]}

        graph: StateGraph[State] = StateGraph(State)
        graph.add_node("work", work)
        graph.set_entry_point("work")
        graph.add_conditional_edges(
            "work",
            lambda s: "done" if s.value >= 4 else "again",
            {"again": "work", "done": END},
        )
        return graph.compile(checkpointer=SqliteCheckpointer(db))

    with pytest.raises(RuntimeError, match="process died"):
        await build().invoke(State(), thread_id="loop")

    store = SqliteCheckpointer(db)
    stored = await store.get("loop")
    assert stored is not None
    stored.status = "running"
    stored.error = None
    await store.put(stored)

    result = await build().resume("loop")

    assert result.state.value == 4
    assert result.state.log == ["pass-1", "pass-2", "pass-3", "pass-4"]


@pytest.mark.asyncio
async def test_resuming_a_completed_thread_returns_its_state_without_rerunning() -> None:
    counter = Counter()
    store = InMemoryCheckpointer()
    compiled = _linear(counter).compile(checkpointer=store)
    first = await compiled.invoke(State(), thread_id="t1")

    again = await compiled.resume("t1")

    assert again.state.label == first.state.label == "done"
    assert again.executed == [], "a completed thread must not re-run anything"
    assert counter.count("a") == 1
    assert counter.count("b") == 1


@pytest.mark.asyncio
async def test_resuming_an_unknown_thread_raises_a_clear_error() -> None:
    store = InMemoryCheckpointer()
    compiled = _linear(Counter()).compile(checkpointer=store)
    await store.put(Checkpoint(thread_id="other"))

    with pytest.raises(UnknownThreadError) as exc:
        await compiled.resume("nope")
    message = str(exc.value)
    assert "'nope'" in message
    assert "other" in message, "the error should list what is actually there"


@pytest.mark.asyncio
async def test_resuming_a_failed_thread_refuses() -> None:
    """A thread marked failed needs a deliberate decision, not a silent retry.

    The failure is staged after a node has completed, because nothing is checkpointed
    before the first node returns — a graph whose *entry* node raises leaves no thread
    behind at all, which the test below covers.
    """
    counter = Counter()

    async def boom(state: State) -> dict[str, Any]:
        raise RuntimeError("boom")

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", value=1))
    graph.add_node("boom", boom)
    graph.set_entry_point("a")
    graph.add_edge("a", "boom")
    graph.add_edge("boom", END)

    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)
    with pytest.raises(RuntimeError, match="boom"):
        await compiled.invoke(State(), thread_id="t1")

    with pytest.raises(RuntimeError, match="checkpointed as failed"):
        await compiled.resume("t1")


# ---------------------------------------------------------------------------
# Resuming a failure on purpose
# ---------------------------------------------------------------------------


def _flaky(counter: Counter, fails: int) -> tuple[StateGraph[State], list[int]]:
    """scrape -> parse -> END, where ``scrape`` raises its first ``fails`` attempts.

    A stage that dies on a network timeout is the case the escape hatch exists for: the
    stages before it completed and are sitting in the checkpoint.
    """
    attempts = [0]

    async def scrape(state: State) -> dict[str, Any]:
        attempts[0] += 1
        if attempts[0] <= fails:
            raise TimeoutError("scrape timed out")
        counter.names.append("scrape")
        return {"label": "scraped"}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("setup", counter.node("setup", value=1))
    graph.add_node("scrape", scrape)
    graph.add_node("parse", counter.node("parse", label="parsed"))
    graph.set_entry_point("setup")
    graph.add_edge("setup", "scrape")
    graph.add_edge("scrape", "parse")
    graph.add_edge("parse", END)
    return graph, attempts


@pytest.mark.asyncio
async def test_acknowledged_failure_resumes_at_the_failed_node() -> None:
    """Completed stages are kept; the run picks up at the one that died."""
    counter = Counter()
    graph, _ = _flaky(counter, fails=1)
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)
    with pytest.raises(TimeoutError):
        await compiled.invoke(State(), thread_id="t1")
    assert counter.count("setup") == 1

    result = await compiled.resume("t1", resume_failed=RESUME_FAILED_ACKNOWLEDGED)

    assert result.state.label == "parsed"
    assert counter.count("setup") == 1, "a completed node must not be re-run"
    assert counter.count("scrape") == 1
    assert counter.count("parse") == 1


@pytest.mark.asyncio
async def test_acknowledged_failure_resumes_a_stream_too() -> None:
    """resume_stream is the same decision, so it takes the same opt-in."""
    from actants.graph.events import GraphCompleted

    counter = Counter()
    graph, _ = _flaky(counter, fails=1)
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)
    with pytest.raises(TimeoutError):
        await compiled.invoke(State(), thread_id="t1")

    with pytest.raises(RuntimeError, match="checkpointed as failed"):
        async for _ in compiled.resume_stream("t1"):
            pass

    seen = [
        event
        async for event in compiled.resume_stream("t1", resume_failed=RESUME_FAILED_ACKNOWLEDGED)
    ]

    assert isinstance(seen[-1], GraphCompleted)
    assert counter.count("setup") == 1, "a completed node must not be re-run"


@pytest.mark.asyncio
async def test_the_original_graph_failure_survives_a_second_one() -> None:
    counter = Counter()
    graph, _ = _flaky(counter, fails=2)
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)
    with pytest.raises(TimeoutError):
        await compiled.invoke(State(), thread_id="t1")

    with pytest.raises(TimeoutError):
        await compiled.resume("t1", resume_failed=RESUME_FAILED_ACKNOWLEDGED)

    stored = await store.get("t1")
    assert stored is not None
    assert stored.status == "failed"
    assert len(stored.prior_errors) == 1, "the first failure is still on record"
    assert counter.count("setup") == 1, "and neither attempt re-ran the completed node"


@pytest.mark.asyncio
async def test_the_graph_opt_in_must_be_spelled_exactly() -> None:
    counter = Counter()
    graph, _ = _flaky(counter, fails=1)
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)
    with pytest.raises(TimeoutError):
        await compiled.invoke(State(), thread_id="t1")

    with pytest.raises(ValueError, match="resume_failed must be exactly"):
        await compiled.resume("t1", resume_failed="yes")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_graph_whose_entry_node_fails_leaves_no_thread() -> None:
    """Nothing is durably recorded before the first node returns.

    Matches ``Agent``: a run that dies before its first checkpoint has no state worth
    resuming, so the thread simply does not exist.
    """

    async def boom(state: State) -> dict[str, Any]:
        raise RuntimeError("boom")

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("boom", boom)
    graph.set_entry_point("boom")
    graph.add_edge("boom", END)

    store = InMemoryCheckpointer()
    with pytest.raises(RuntimeError, match="boom"):
        await graph.compile(checkpointer=store).invoke(State(), thread_id="t1")

    assert await store.get("t1") is None


@pytest.mark.asyncio
async def test_resume_without_a_checkpointer_is_rejected() -> None:
    compiled = _linear(Counter()).compile()
    with pytest.raises(ValueError, match="checkpointer"):
        await compiled.resume("t1")


@pytest.mark.asyncio
async def test_resuming_an_agent_thread_as_a_graph_is_refused() -> None:
    """One store may hold both kinds of run; misreading one as the other is a bug."""
    store = InMemoryCheckpointer()
    await store.put(Checkpoint(thread_id="agent-thread", status="running"))

    compiled = _linear(Counter()).compile(checkpointer=store)
    with pytest.raises(GraphValidationError, match="Agent checkpoint"):
        await compiled.resume("agent-thread")


@pytest.mark.asyncio
async def test_concurrent_durable_runs_keep_separate_threads() -> None:
    """One compiled graph driving two runs must not mix their state or their results."""

    async def slow(state: State) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"value": state.value + 1}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("slow", slow)
    graph.set_entry_point("slow")
    graph.add_edge("slow", END)

    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store)

    first, second = await asyncio.gather(
        compiled.invoke(State(value=10), thread_id="t1"),
        compiled.invoke(State(value=20), thread_id="t2"),
    )

    assert first.state.value == 11
    assert second.state.value == 21
    assert first.executed == ["slow"], "one run's executed list must not collect the other's"
    assert second.executed == ["slow"]
    assert sorted(await store.list_threads()) == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------


def _guarded(counter: Counter, store: Any) -> CompiledGraph[State]:
    """a -> guarded -> END, pausing before ``guarded``."""
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("a", counter.node("a", value=1))
    graph.add_node("guarded", counter.node("guarded", label="sent"))
    graph.set_entry_point("a")
    graph.add_edge("a", "guarded")
    graph.add_edge("guarded", END)
    return graph.compile(checkpointer=store, interrupt_before=["guarded"])


@pytest.mark.asyncio
async def test_interrupt_pauses_before_the_node_runs() -> None:
    counter = Counter()
    store = InMemoryCheckpointer()
    compiled = _guarded(counter, store)

    result = await compiled.invoke(State(), thread_id="t1")

    assert result.interrupted is True
    assert result.pending_node == "guarded"
    assert result.thread_id == "t1"
    assert result.executed == ["a"]
    assert counter.count("guarded") == 0, "the guarded node must not have run"

    checkpoint = await store.get("t1")
    assert checkpoint is not None
    assert checkpoint.status == "interrupted"


@pytest.mark.asyncio
async def test_interrupt_then_approve_runs_the_node_and_continues() -> None:
    counter = Counter()
    store = InMemoryCheckpointer()
    compiled = _guarded(counter, store)
    await compiled.invoke(State(), thread_id="t1")

    result = await compiled.resume("t1", approve=True)

    assert result.interrupted is False
    assert result.state.label == "sent"
    assert result.state.value == 1, "state from before the pause must carry through"
    assert counter.count("guarded") == 1
    assert counter.count("a") == 1, "the node before the pause must not re-run"
    assert result.executed == ["guarded"]


@pytest.mark.asyncio
async def test_interrupt_then_reject_skips_the_node_and_continues() -> None:
    counter = Counter()
    store = InMemoryCheckpointer()
    compiled = _guarded(counter, store)
    await compiled.invoke(State(), thread_id="t1")

    result = await compiled.resume("t1", approve=False)

    assert result.interrupted is False
    assert result.state.label == "", "a rejected node must not have applied its update"
    assert counter.count("guarded") == 0, "a rejected node must not run"
    assert counter.count("a") == 1


@pytest.mark.asyncio
async def test_rejecting_a_node_still_runs_the_rest_of_the_graph() -> None:
    """Rejection routes past the node rather than ending the run."""
    counter = Counter()
    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("guarded", counter.node("guarded", label="sent"))
    graph.add_node("after", counter.node("after", value=42))
    graph.set_entry_point("guarded")
    graph.add_edge("guarded", "after")
    graph.add_edge("after", END)

    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store, interrupt_before=["guarded"])
    await compiled.invoke(State(), thread_id="t1")

    result = await compiled.resume("t1", approve=False)

    assert counter.count("guarded") == 0
    assert result.state.value == 42, "rejecting one node must not abandon the rest"


@pytest.mark.asyncio
async def test_a_guarded_node_in_a_loop_pauses_on_every_pass() -> None:
    """Approving once must not disarm the guard for the rest of the run.

    The regression this pins: the flag that lets an approved node through has to be
    consumed by that one node, not left set for every later visit.
    """
    counter = Counter()

    async def work(state: State) -> dict[str, Any]:
        counter.names.append("work")
        return {"value": state.value + 1}

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("work", work)
    graph.set_entry_point("work")
    graph.add_conditional_edges(
        "work",
        lambda s: "done" if s.value >= 2 else "again",
        {"again": "work", "done": END},
    )

    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store, interrupt_before=["work"])

    first = await compiled.invoke(State(), thread_id="t1")
    assert first.interrupted and counter.count("work") == 0

    second = await compiled.resume("t1", approve=True)
    assert second.interrupted is True, "the second pass must pause again"
    assert second.pending_node == "work"
    assert counter.count("work") == 1

    third = await compiled.resume("t1", approve=True)
    assert third.interrupted is False
    assert third.state.value == 2
    assert counter.count("work") == 2


@pytest.mark.asyncio
async def test_resuming_an_interrupted_thread_without_a_decision_is_an_error() -> None:
    compiled = _guarded(Counter(), InMemoryCheckpointer())
    await compiled.invoke(State(), thread_id="t1")

    with pytest.raises(ValueError) as exc:
        await compiled.resume("t1")
    message = str(exc.value)
    assert "approve=True" in message
    assert "guarded" in message


@pytest.mark.asyncio
async def test_interrupt_survives_a_fresh_sqlite_checkpointer(tmp_path: Path) -> None:
    """HITL must work across processes: the pending node lives in the file."""
    db = tmp_path / "runs.db"
    counter = Counter()

    paused = await _guarded(counter, SqliteCheckpointer(db)).invoke(State(), thread_id="job-7")
    assert paused.interrupted
    assert counter.count("guarded") == 0

    result = await _guarded(counter, SqliteCheckpointer(db)).resume("job-7", approve=True)

    assert result.state.label == "sent"
    assert counter.count("guarded") == 1
    assert counter.count("a") == 1, "the earlier node must not re-run in the new process"


@pytest.mark.asyncio
async def test_interrupt_without_a_thread_id_is_an_error() -> None:
    """There would be nothing to resume from, so refuse rather than silently run."""
    counter = Counter()
    compiled = _guarded(counter, InMemoryCheckpointer())

    with pytest.raises(ValueError, match="interrupt_before"):
        await compiled.invoke(State())
    assert counter.count("guarded") == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_an_event_per_node_and_ends_completed() -> None:
    from actants.graph.events import GraphCompleted, GraphNodeCompleted, GraphNodeStarted

    compiled = _linear(Counter()).compile()
    events = [e async for e in compiled.stream(State())]

    assert [type(e).__name__ for e in events] == [
        "GraphNodeStarted",
        "GraphNodeCompleted",
        "GraphNodeStarted",
        "GraphNodeCompleted",
        "GraphCompleted",
    ]
    started = [e for e in events if isinstance(e, GraphNodeStarted)]
    assert [e.node for e in started] == ["a", "b"]
    completed = [e for e in events if isinstance(e, GraphNodeCompleted)]
    assert completed[0].state.value == 1, "each event carries the state as of that node"
    final = events[-1]
    assert isinstance(final, GraphCompleted)
    assert final.state.label == "done"


@pytest.mark.asyncio
async def test_stream_ends_in_an_interrupted_event_when_paused() -> None:
    from actants.graph.events import GraphInterrupted

    compiled = _guarded(Counter(), InMemoryCheckpointer())
    events = [e async for e in compiled.stream(State(), thread_id="t1")]

    final = events[-1]
    assert isinstance(final, GraphInterrupted)
    assert final.node == "guarded"


# ---------------------------------------------------------------------------
# Agent as a node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agent_is_usable_as_a_node() -> None:
    """The two halves of the framework must compose without ceremony."""
    agent = Agent(
        llm=LLM(provider=FakeLLMProvider([fake_completion("42")]), model="m", tracing=False)
    )

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node(
        "ask",
        agent_node(agent, prompt=lambda s: f"answer: {s.label}", output="label"),
    )
    graph.set_entry_point("ask")
    graph.add_edge("ask", END)

    result = await graph.compile().invoke(State(label="what is 6*7"))

    assert result.state.label == "42"
    assert result.executed == ["ask"]


@pytest.mark.asyncio
async def test_an_agent_node_accumulates_into_an_append_field() -> None:
    """An agent inside a loop: each pass adds an answer rather than replacing it."""
    agent = Agent(
        llm=LLM(
            provider=FakeLLMProvider([fake_completion("first"), fake_completion("second")]),
            model="m",
            tracing=False,
        )
    )

    graph: StateGraph[State] = StateGraph(State)
    graph.add_node("ask", agent_node(agent, prompt=lambda s: "go", output="log"))
    graph.add_node("count", Counter().node("count"))
    graph.set_entry_point("ask")
    graph.add_edge("ask", "count")
    graph.add_conditional_edges(
        "count",
        lambda s: "done" if len(s.log) >= 2 else "again",
        {"again": "ask", "done": END},
    )

    result = await graph.compile().invoke(State())

    assert result.state.log == ["first", "second"]


@pytest.mark.asyncio
async def test_an_agent_node_is_durable_like_any_other(tmp_path: Path) -> None:
    """A graph whose node is an Agent resumes without re-running that agent turn."""
    db = tmp_path / "runs.db"
    provider = FakeLLMProvider([fake_completion("answered")])
    agent = Agent(llm=LLM(provider=provider, model="m", tracing=False))
    calls = {"after": 0}

    def build(*, explode: bool) -> CompiledGraph[State]:
        async def after(state: State) -> dict[str, Any]:
            calls["after"] += 1
            if explode:
                raise RuntimeError("process died")
            return {"value": 1}

        graph: StateGraph[State] = StateGraph(State)
        graph.add_node("ask", agent_node(agent, prompt=lambda s: "go", output="label"))
        graph.add_node("after", after)
        graph.set_entry_point("ask")
        graph.add_edge("ask", "after")
        graph.add_edge("after", END)
        return graph.compile(checkpointer=SqliteCheckpointer(db))

    with pytest.raises(RuntimeError, match="process died"):
        await build(explode=True).invoke(State(), thread_id="job")

    store = SqliteCheckpointer(db)
    stored = await store.get("job")
    assert stored is not None
    stored.status = "running"
    stored.error = None
    await store.put(stored)

    result = await build(explode=False).resume("job")

    assert result.state.label == "answered", "the agent's answer survived the crash"
    assert result.state.value == 1
    assert result.executed == ["after"], "the agent node must not run a second turn"
    # The fake provider was scripted with exactly one completion; a second agent turn
    # would have exhausted it.


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "StateGraph",
        "CompiledGraph",
        "GraphResult",
        "END",
        "Append",
        "EndT",
        "NodeFn",
        "RouterFn",
        "agent_node",
        "GraphEvent",
        "GraphNodeStarted",
        "GraphNodeCompleted",
        "GraphInterrupted",
        "GraphCompleted",
        "GraphError",
        "GraphValidationError",
        "GraphRecursionError",
    ],
)
def test_graph_symbols_are_public(name: str) -> None:
    import actants

    assert name in actants.__all__
    assert getattr(actants, name) is not None


@pytest.mark.parametrize("name", ["GraphError", "GraphValidationError", "GraphRecursionError"])
def test_graph_errors_join_the_hierarchy(name: str) -> None:
    import actants

    assert issubclass(getattr(actants, name), ActantsError)


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------

#: A consumer that mistypes its nodes in the three ways that matter. Each marked line
#: must produce an error; the unmarked ones must not.
_MISTYPED_CONSUMER = """
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from actants import END, StateGraph


class State(BaseModel):
    question: str = ""


class Other(BaseModel):
    unrelated: int = 0


async def good(s: State) -> dict[str, Any]:
    return {"question": s.question}


async def wrong_state(s: Other) -> dict[str, Any]:
    return {}


def router(s: State) -> str:
    return "done"


def wrong_router(s: Other) -> str:
    return "done"


graph: StateGraph[State] = StateGraph(State)
graph.add_node("good", good)
graph.add_node("bad", wrong_state)
graph.add_conditional_edges("good", router, {"done": END})
graph.add_conditional_edges("good", wrong_router, {"done": END})


async def main() -> None:
    compiled = graph.compile()
    result = await compiled.invoke(State())
    wrong: int = result.state.question
"""


def _have_mypy() -> bool:
    return importlib.util.find_spec("mypy") is not None or shutil.which("mypy") is not None


def _run_mypy(script: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Type-check ``script`` with the mypy belonging to the running interpreter.

    Invoked as ``-m mypy`` rather than the bare executable: in a venv-based checkout
    mypy is installed but not on PATH, so a ``shutil.which`` lookup finds nothing and
    the guard tests below would skip silently forever — which is the one failure mode a
    type-safety test must not have.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy"),
            str(script),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )


@pytest.mark.skipif(not _have_mypy(), reason="mypy not installed")
def test_a_mistyped_node_is_caught_by_mypy_strict(tmp_path: Path) -> None:
    """The type parameters must be real, not decorative.

    A graph whose nodes are only checked at runtime is barely better than dicts, so this
    pins the three failures that must be compile-time errors: a node and a router written
    against the wrong state model, and a field read at the wrong type off the result.
    """
    script = tmp_path / "mistyped.py"
    script.write_text(_MISTYPED_CONSUMER, encoding="utf-8")
    proc = _run_mypy(script, tmp_path)
    if proc.returncode == 2 and "INTERNAL ERROR" in proc.stderr:
        pytest.skip(f"mypy crashed internally (not an actants failure): {proc.stderr[-300:]}")

    output = proc.stdout
    assert 'Argument 2 to "add_node"' in output, (
        f"a node written against the wrong state model was not rejected:\n{output}"
    )
    assert 'Argument 2 to "add_conditional_edges"' in output, (
        f"a router written against the wrong state model was not rejected:\n{output}"
    )
    # GraphResult.state must be the caller's concrete model, not Any — otherwise every
    # downstream field access silently type-checks.
    assert "Incompatible types in assignment" in output, (
        f"result.state degraded to Any, so field types are unchecked:\n{output}"
    )


@pytest.mark.skipif(not _have_mypy(), reason="mypy not installed")
def test_a_correctly_typed_graph_passes_mypy_strict(tmp_path: Path) -> None:
    """The other half: correct usage must not need a single ignore comment."""
    script = tmp_path / "typed_ok.py"
    script.write_text(
        _MISTYPED_CONSUMER.replace('graph.add_node("bad", wrong_state)\n', "")
        .replace('graph.add_conditional_edges("good", wrong_router, {"done": END})\n', "")
        .replace(
            "    wrong: int = result.state.question\n", "    ok: str = result.state.question\n"
        ),
        encoding="utf-8",
    )
    proc = _run_mypy(script, tmp_path)
    if proc.returncode == 2 and "INTERNAL ERROR" in proc.stderr:
        pytest.skip(f"mypy crashed internally (not an actants failure): {proc.stderr[-300:]}")
    assert proc.returncode == 0, (
        f"a correctly typed graph consumer failed mypy --strict:\n{proc.stdout}\n{proc.stderr}"
    )


def test_graph_errors_keep_their_builtin_bases() -> None:
    """Existing `except ValueError` / `except RuntimeError` handlers must still fire."""
    assert issubclass(GraphValidationError, ValueError)
    assert issubclass(GraphRecursionError, RuntimeError)
    assert issubclass(GraphValidationError, GraphError)
    assert issubclass(GraphRecursionError, GraphError)
