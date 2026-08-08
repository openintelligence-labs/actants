# StateGraph

`Agent` is a linear loop: think, call tools, repeat until done. `StateGraph` is for the
shapes that loop cannot express — a router that picks one of three branches, a critic
that sends work back for another pass, a pipeline whose stages each need their own
prompt.

The state is a pydantic model you define. Nodes receive it and return a partial update;
edges say what runs next. Nothing runs until you `compile()`, which is where the
structural checks happen.

## A first graph

<!-- docs-test: run -->

```python
from typing import Any

from pydantic import BaseModel

from actants import END, StateGraph


class State(BaseModel):
    question: str
    draft: str = ""


async def write(state: State) -> dict[str, Any]:
    return {"draft": f"a draft answering {state.question}"}


async def polish(state: State) -> dict[str, Any]:
    return {"draft": state.draft.upper()}


graph: StateGraph[State] = StateGraph(State)
graph.add_node("write", write)
graph.add_node("polish", polish)
graph.set_entry_point("write")
graph.add_edge("write", "polish")
graph.add_edge("polish", END)

result = await graph.compile().invoke(State(question="why?"))
assert result.state.draft.startswith("A DRAFT")
assert result.executed == ["write", "polish"]
```

`END` is the terminal node name. Point an edge at it — or have a router return it — to
finish the run. `add_node`, `add_edge`, `add_conditional_edges`, and `set_entry_point`
all return the graph, so this can be written as one chain if you prefer.

`invoke()` returns a `GraphResult` with the final `state`, the list of nodes this call
`executed`, and the `interrupted` / `pending_node` / `thread_id` fields covered below.

## What a node returns

Three forms, whichever fits:

| Return | Meaning |
|---|---|
| `dict[str, Any]` | A partial update: only the named fields change. |
| the state model | Replaces the state wholesale. |
| `None` | Nothing changes. For a node that only performs a side effect. |

A dict key that is not a field of the state model raises `GraphValidationError` naming
the node. That check exists because a type checker cannot see inside a returned dict, so
this is the only place a typo like `{"anwser": ...}` can be caught — and silently
dropping it would surface much later as a field that mysteriously never changed.

Merging never mutates the state a node was given, and never shares a mutable container
with it: the merged state is independent, so a node that mutates a list in place cannot
reach back into the value an earlier node — or a concurrent run — is holding. A node that
raises leaves the state exactly as the last successful node left it.

`invoke()` deep-copies the state you pass it for the same reason, so one seed object can
safely start several concurrent runs.

## Reducers: accumulating instead of replacing

A field updates by replacement by default, which is what plain assignment would do. A
graph that loops almost always has one field that should accumulate instead — a message
list, a scratchpad of findings. Annotate it with `Append`:

<!-- docs-test: run -->

```python
from typing import Annotated, Any

from pydantic import BaseModel, Field

from actants import END, Append, StateGraph


class State(BaseModel):
    findings: Annotated[list[str], Append] = Field(default_factory=list)
    passes: int = 0


async def research(state: State) -> dict[str, Any]:
    return {"findings": [f"finding {state.passes + 1}"], "passes": state.passes + 1}


graph: StateGraph[State] = StateGraph(State)
graph.add_node("research", research)
graph.set_entry_point("research")
graph.add_conditional_edges(
    "research",
    lambda s: "done" if s.passes >= 3 else "again",
    {"again": "research", "done": END},
)

result = await graph.compile().invoke(State())
assert result.state.findings == ["finding 1", "finding 2", "finding 3"]
assert result.state.passes == 3
```

`passes` replaces; `findings` extends. A node returning a bare value for an `Append`
field (`{"findings": "one"}`) is taken as a single item rather than being spread into
characters.

`Append` on a field that is not a list is rejected when the `StateGraph` is constructed,
not on the first update — a scalar marked `Append` is a mistake in the model definition,
and there is no reason to let a graph run for several nodes before saying so.

## Conditional edges

A router is a **sync** function returning a key of the mapping given alongside it:

```python
def route(state: State) -> str:
    if state.score >= 8:
        return "ship"
    return "revise"


graph.add_conditional_edges("critique", route, {"ship": END, "revise": "write"})
```

The indirection through the mapping is what makes the branch inspectable: `compile()` can
check that every reachable target exists, which it could not do if the router returned
node names directly. A router returning a key that is not in the mapping raises
`GraphValidationError` listing the keys that are.

Routers are sync deliberately. A router that needs to await is doing work, and work
belongs in a node where it gets checkpointed — a router runs *after* its node's state is
durably recorded and must be cheap enough to simply re-run on resume.

That ordering is what makes a raising router safe: the node's completion is already in the
checkpoint, so a router that raises — an unmapped key, a bug — fails the run without
un-recording work that already happened. Resume picks up at the routing decision and
re-runs only the router, never the node.

A node has an unconditional edge or conditional edges, never both.

## Loops and `max_iterations`

Point a branch back at an earlier node and the graph loops. There is no structural way to
tell a slow convergence from an infinite one, so `compile()` takes a budget:

```python
compiled = graph.compile(max_iterations=25)  # the default
```

It counts nodes executed in one run. Exceeding it raises `GraphRecursionError`, which
carries `.node` and `.iterations` and names the node that was about to run again — the
router that keeps selecting it is where to look.

## Compile-time validation

`compile()` rejects, before anything can run:

- no entry point, or one that is not a registered node
- an edge or conditional target naming a node that does not exist
- a node with no outgoing edge (the run would stop there without reaching `END`)
- a node unreachable from the entry point
- an `interrupt_before` naming a node the graph does not have

The unreachability check is the one that earns its keep: an orphaned node is dead code
that reads as live code, and is nearly always a typo in an edge or a branch someone forgot
to wire up.

## Streaming

`stream()` yields an event as each node starts and finishes:

```python
from actants import GraphCompleted, GraphInterrupted, GraphNodeCompleted, GraphNodeStarted

async for event in compiled.stream(State(question="why?")):
    match event:
        case GraphNodeStarted(node=name):
            print(f"→ {name}")
        case GraphNodeCompleted(node=name, state=state):
            print(f"← {name}")
        case GraphInterrupted(node=name):
            print(f"paused before {name}")
        case GraphCompleted(state=state):
            print("done")
```

Every run ends in exactly one of the two terminal events, `GraphInterrupted` or
`GraphCompleted`. `resume_stream()` is the streaming form of `resume()` and yields the
same events.

## Durability

Pass a checkpointer to `compile()` and a `thread_id` to `invoke()`, and the state is
persisted after *every* node completes:

```python
from actants import SqliteCheckpointer

compiled = graph.compile(checkpointer=SqliteCheckpointer("runs.db"))
result = await compiled.invoke(State(question="why?"), thread_id="job-1")
# ...process dies...
result = await compiled.resume("job-1")
```

**The guarantee: every node whose completion was recorded is skipped, never re-run.** The
run picks up at the node the checkpoint says was next, with the state exactly as the last
completed node left it. This is the same at-most-once guarantee
[`Agent.resume()`](durability.md#the-guarantee) gives for completed tool calls, applied
at node granularity.

The boundary is different from the agent's, and simpler: a node either completed and was
recorded, or it did not. A node that was *running* when the process died is re-run from
the start on resume, because its update never landed. That makes a node the unit of
at-least-once — so a node containing an unrepeatable side effect should be a node that
contains only that side effect, keeping the replayed blast radius as small as possible.
actants has no per-node `idempotent` flag; the agent's `idempotent=False` applies to tool
calls, not to graph nodes.

`resume()` on a thread that already completed returns its stored state without running
anything, and `executed` comes back empty. An unknown thread id raises
`UnknownThreadError`; a thread marked failed raises `RuntimeError` rather than silently
retrying. A graph whose *entry* node fails leaves no thread at all — nothing is
checkpointed before the first node returns.

A failed thread can still be resumed deliberately, which is what you want when one stage
of a long pipeline died on a network timeout and the stages before it are sitting in the
checkpoint:

```python
from actants import RESUME_FAILED_ACKNOWLEDGED

result = await compiled.resume("job-1", resume_failed=RESUME_FAILED_ACKNOWLEDGED)
```

`resume_stream()` takes the same argument. The run picks up at the node the checkpoint
says was next and every completed node is skipped, exactly as for a crashed thread — the
opt-in changes whether the run is allowed to continue, not which nodes it runs. Only the
node that was executing when the failure hit is re-run, so the
[at-least-once caveat](#durability) above is the one to think about before typing it: if
that node writes somewhere, it writes again. The failure being resumed past moves to
`checkpoint.prior_errors` rather than being overwritten, so it is still readable if the
resumed run fails too. See
[Resuming a failure anyway](durability.md#resuming-a-failure-anyway) for when this is and
is not a safe thing to do.

Graph runs reuse the agent's `Checkpointer` protocol, so one store — and one SQLite file
— holds both. They are tagged, so resuming a graph thread with `Agent.resume` (or an
agent thread with `CompiledGraph.resume`) fails loudly instead of misreading the payload.

Two processes resuming the same thread id concurrently is undefined; actants does not
lock a thread across processes.

## Interrupts

`interrupt_before` names nodes the graph must not run on its own:

```python
compiled = graph.compile(
    checkpointer=SqliteCheckpointer("runs.db"),
    interrupt_before=["send_email"],
)

result = await compiled.invoke(State(question="why?"), thread_id="job-1")
if result.interrupted:
    print(f"paused before {result.pending_node}")

await compiled.resume("job-1", approve=True)  # run it and continue
await compiled.resume("job-1", approve=False)  # skip it and route onward
```

`result.state` holds the state as it stood *before* the guarded node. `approve=False`
treats the node as having run and changed nothing, so the run continues past it rather
than dying.

A guarded node inside a loop pauses on every pass; approving once does not disarm the
guard. Resuming an interrupted thread without `approve` raises `ValueError`, and hitting
a guarded node with no checkpointer and `thread_id` raises too — there would be nothing
to resume from.

## An Agent as a node

`agent_node` adapts an `Agent` into a node function. `prompt` reads the state to build the
agent's user turn; `output` names the state field its final answer lands in:

<!-- docs-test: run -->

```python
from pydantic import BaseModel

from actants import END, Agent, LLM, StateGraph, agent_node
from actants.testing import FakeLLMProvider, fake_completion


class State(BaseModel):
    question: str
    answer: str = ""


agent = Agent(
    llm=LLM(
        provider=FakeLLMProvider([fake_completion("42")]),
        model="fake",
        tracing=False,
    )
)

graph: StateGraph[State] = StateGraph(State)
graph.add_node("ask", agent_node(agent, prompt=lambda s: s.question, output="answer"))
graph.set_entry_point("ask")
graph.add_edge("ask", END)

result = await graph.compile().invoke(State(question="what is 6*7?"))
assert result.state.answer == "42"
assert result.executed == ["ask"]
```

If `output` names a field annotated `Append`, the answers accumulate across passes rather
than replacing each other — which is what makes an agent usable inside a retry or
critique cycle.

The agent's own `thread_id` is deliberately *not* wired to the graph's. The graph
checkpoints the state after the node completes, and that is the boundary that matters for
"this node already ran"; giving the inner agent its own durable thread as well would make
a resumed graph replay a half-finished agent turn inside a node the graph has already
recorded as done. If you want the inner loop durable too, pass a checkpointed `Agent`
explicitly and derive its thread id from your own state.

## Evaluating a graph

An `EvalSuite` runs against a `CompiledGraph` as readily as against an `Agent`; the case
input is a state instance rather than a prompt string. See [Testing](testing.md).

## Limits

- Nodes run **one at a time**. There is no parallel fan-out; a graph is a sequence of
  node executions chosen by its edges.
- The state must be JSON-serializable to be checkpointed, since it round-trips through
  `model_dump_json` / `model_validate_json`.
- State is validated on the way *back* from a checkpoint but not re-validated on every
  merge, so a field validator does not run per node. Values come from typed node
  functions; unknown keys are still rejected.
