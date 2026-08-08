# Durability

A long agent run has side effects. It charges a card, sends an email, files a ticket.
If the process dies half way through, the two bad outcomes are losing the work and
doing the work twice — and the second is usually worse.

A `Checkpointer` plus a `thread_id` makes a run resumable. State is persisted after
every LLM completion and after *each individual tool result*, so `resume()` can pick the
run back up without paying for the completions again and without re-running the tools
that already finished.

Durability is opt-in per run. An `Agent` built without a `checkpointer`, or a `run()`
called without a `thread_id`, touches no storage and behaves exactly as it did before
any of this existed.

## Making a run resumable

```python
from actants import Agent, LLM, SqliteCheckpointer, ToolRegistry

tools = ToolRegistry()

agent = Agent(
    llm=LLM(model="llama3.2"),
    tools=tools,
    checkpointer=SqliteCheckpointer("runs.db"),
)

result = await agent.run("book the flight", thread_id="job-42")
```

If that process dies, a new one resumes the thread by reading the same file:

```python
from actants import Agent, LLM, SqliteCheckpointer, ToolRegistry

agent = Agent(
    llm=LLM(model="llama3.2"),
    tools=ToolRegistry(),
    checkpointer=SqliteCheckpointer("runs.db"),
)
result = await agent.resume("job-42")
```

Nothing is cached in the `SqliteCheckpointer` object, which is the point: the resuming
process reads exactly what the crashed one committed to the file. `thread_id` without a
checkpointer raises `ValueError`; a checkpointer with no `thread_id` simply persists
nothing.

## The guarantee

This is the part worth reading carefully, because it is narrower than "exactly once" and
saying otherwise would be a lie.

**Resume is at-most-once for every tool call whose result was recorded before the crash.**
Those calls are replayed from the checkpoint and never dispatched again. A step with
three tool calls that died after the second one comes back and dispatches only the third.

**Resume is at-least-once for the single call that was in flight when the process died.**
That call is the irreducible ambiguity. actants wrote a checkpoint before dispatching it
and would have written another once the result landed; the process died between the two.
From the outside, "the request never left" and "the request was handled and the reply was
lost" look identical, so there is nothing in the durable record that can distinguish
them. No amount of checkpointing closes that gap — closing it requires the *tool's* side
of the call to be idempotent, or to carry an idempotency key the vendor deduplicates on.

So actants does not pretend to know. It applies the tool's own declaration:

- `idempotent=True` (the default) — the call is re-dispatched. This is the at-least-once
  half.
- `idempotent=False` — the call is not re-dispatched. `resume()` raises
  `UnresolvedToolCallError` instead, carrying the thread id and the `ToolCall`, so the
  caller can decide.
- **A tool the resuming agent's registry does not have** — renamed or removed since the
  run started — is treated as `idempotent=False`. An unknown tool is precisely the case
  where actants cannot establish that repeating it is safe, so it raises rather than
  guessing.

Exactly one call per thread is ever ambiguous. Everything before it is settled.

**Concurrent `resume()` of one thread id is serialized within a process.** Two `resume()`
calls on one `Agent` for the same thread id queue on an in-process lock, so the
read-decide-dispatch sequence cannot interleave: the second runs after the first has
committed, sees a completed thread, and returns its stored result without dispatching
anything. The at-most-once half of the guarantee therefore holds for them.

This lock is per thread id, so unrelated threads still resume concurrently — and it is an
`asyncio` lock in one process. **Two *processes* resuming the same thread id concurrently
remains undefined**, exactly as before; no in-process lock can close that, and arranging
that exclusion is still yours.

## Declaring what is unsafe to repeat

`idempotent` defaults to `True` because most tools are reads, and that default is wrong
for anything that writes. Set it explicitly on those:

<!-- docs-test: run -->

```python
from actants import ToolRegistry

tools = ToolRegistry()


async def lookup_order(order_id: str) -> str:
    return f"order {order_id}"


async def charge_card(order_id: str, cents: int) -> str:
    return f"charged {cents}"


tools.register_function("lookup_order", "Look an order up", lookup_order)
tools.register_function(
    "charge_card",
    "Charge the card on file",
    charge_card,
    idempotent=False,
)
```

## Resolving an unresolved call

When `resume()` raises `UnresolvedToolCallError`, the call id in the exception is the same
id the tool saw. If the vendor supports an idempotency key, or you keep an outbox, that
id is what you look the call up by. Then resume again with a decision:

```python
from actants import UnresolvedToolCallError

try:
    result = await agent.resume("job-42")
except UnresolvedToolCallError as exc:
    print(f"{exc.call.name} (id {exc.call.id}) may or may not have run")
    # Established it did not run:
    result = await agent.resume("job-42", resolve="retry")
```

The three resolutions:

| `resolve=` | Effect |
|---|---|
| `"abort"` (default) | Raise `UnresolvedToolCallError`. No tool is dispatched. |
| `"retry"` | Dispatch the call anyway. Use once you have established it did not run. |
| `"skip"` | Record a tool result saying the call was not re-run and its outcome is unknown, then let the model continue. |

`"skip"` writes an ordinary-looking failed tool result into the history rather than a
special marker, so the model reacts to it the way it reacts to any tool error. That
leaves the model working from an incomplete picture — it is the right choice when the
call is unsafe to repeat and the run can proceed without it, not a way to make the
ambiguity go away.

`resolve` applies only to an in-flight call. It is ignored otherwise.

## Interrupts (human in the loop)

`interrupt_before` names tools the agent must not dispatch on its own. When the model
asks for one, the run persists itself and stops in front of the call:

```python
from actants import Agent, LLM, SqliteCheckpointer, ToolRegistry

agent = Agent(
    llm=LLM(model="llama3.2"),
    tools=ToolRegistry(),
    checkpointer=SqliteCheckpointer("runs.db"),
    interrupt_before=["send_email"],
)

result = await agent.run("email the customer an apology", thread_id="job-7")
if result.interrupted:
    print(f"waiting on approval for {result.pending_call.name}({result.pending_call.arguments})")
```

`result.final` holds the completion that *asked* for the call — the model's last word
before it was paused. The turn is not committed to the agent's memory while it is paused,
exactly as a failed turn is not.

A decision resumes it:

```python
approved = await agent.resume("job-7", approve=True)  # dispatch it and carry on
rejected = await agent.resume("job-7", approve=False)  # record a refusal and carry on
```

`approve=False` appends a tool result saying a human rejected the call, so the model gets
to respond to the refusal rather than the run dying. Any remaining calls in that step
still run.

Because the pending call lives in the checkpoint, the approval can come from a different
process entirely — an HTTP handler, a Slack action, a cron job the next morning.
Resuming an interrupted thread without passing `approve` raises `ValueError`: there is a
decision to make and actants will not make it for you.

Two details worth knowing:

- A guarded tool that comes around again in a later step pauses again. Approving once
  does not disarm the guard for the rest of the run.
- `interrupt_before` with no checkpointer and `thread_id` raises when the pause is
  reached, because there would be nothing to resume from.

## Thread states

A checkpoint records where the run stood, and `resume()` behaves differently for each:

| Status | `resume()` does |
|---|---|
| `running` | Continues the run. |
| `interrupted` | Needs `approve=True` / `approve=False`; raises `ValueError` without one. |
| `completed` | Returns the stored result without re-running anything. |
| `failed` | Raises `RuntimeError` describing the original failure, unless you opt in. |

An unknown `thread_id` raises `UnknownThreadError`, listing the threads the store does
hold.

A `failed` thread is deliberately not auto-resumable. The failure may have come from a
torn turn, and a silent retry of an unknown failure is how a side effect gets repeated.
Inspect it, and start a new run or delete the thread. Note that a run which dies before
its first checkpoint leaves no thread at all — there is no state worth resuming.

### Resuming a failure anyway

The default above is right for the general case, but it is not the whole story: an
ordinary exception — a network timeout in one stage of a long run — is far more common
than a process that vanishes, and the completed work is provably still in the checkpoint.
Refusing outright means an operator has no way to recover it.

So there is an escape hatch, and it is deliberately awkward to type:

```python
from actants import RESUME_FAILED_ACKNOWLEDGED

result = await agent.resume("job-42", resume_failed=RESUME_FAILED_ACKNOWLEDGED)
```

The constant's value is the sentence `"i-know-the-failure-may-have-half-run"`. It is not
a boolean, and nothing but that exact string is accepted, so it cannot be set by a
generic retry wrapper forwarding flags it does not understand, and it cannot be reached
by a truthy default. Typing it is a statement about a specific thread.

**What it does not do.** It does not loosen the at-most-once guarantee by one inch. A
thread that failed *while a tool was in flight* still goes through `resolve`, so a call
registered `idempotent=False` still raises `UnresolvedToolCallError` — opting in to
resume a failure is not opting in to replay an unsafe side effect, and those are separate
decisions because they are answered by separate evidence. Every tool call whose result
was recorded is replayed from the checkpoint, never dispatched again.

**When it is safe.** When you know what the failure was and that it left nothing
half-done — a timeout on a read, a transient dependency outage, an exception thrown
between side effects. Read `checkpoint.error` first; that is the point of keeping it.

**When it is not.** When the failure is unexplained, when it came from a tool that writes
somewhere, or when the answer is "let's just retry it and see". The refusal is the
correct outcome in all three, and re-running from scratch is cheaper than a duplicated
charge.

A thread resumed this way goes back to `running` and, if it gets there, `completed` — it
is genuinely running again, and leaving it marked `failed` would make the status a lie.
The failure it was resumed past is not lost: it moves to `checkpoint.prior_errors`, which
accumulates across resumes. If the resumed run fails again the new failure lands in
`error` and the original stays in `prior_errors`, so a thread that has been retried
several times still shows what went wrong the first time.

## Checkpointer implementations

`InMemoryCheckpointer` is process-local. It survives a failed *run*, not a dead
*process*, which makes it right for tests and for approve/reject flows inside one
process.

`SqliteCheckpointer` writes to a file and is what durability across a restart means.
Each call opens its own connection, so the checkpointer instance holds no state and
separate processes writing different thread ids coexist on WAL.

<!-- docs-test: run -->

```python
from actants import Checkpoint, InMemoryCheckpointer

store = InMemoryCheckpointer()
await store.put(Checkpoint(thread_id="job-42", status="running"))

loaded = await store.get("job-42")
assert loaded is not None and loaded.status == "running"
assert await store.list_threads() == ["job-42"]
assert await store.delete("job-42") is True
```

`Checkpointer` is a runtime-checkable `Protocol` with four async methods — `put`, `get`,
`list_threads`, `delete` — so a store backed by Redis, Postgres, or an object store is
just a class implementing them. `put` must be a full overwrite of the thread's state, not
an append: the agent always hands over the complete `Checkpoint`, and resume only ever
reads the latest one.

Concurrent access to *different* thread ids must be safe. Two writers on the *same*
thread id is undefined, and actants does not lock a thread across processes — if two
workers might resume the same thread, that exclusion is yours to arrange.

## Schema versions

A `SqliteCheckpointer` file records its schema version in `PRAGMA user_version`. Opening
one written by an incompatible actants raises `CheckpointSchemaMismatch` rather than
resetting the file.

That is deliberately unlike the semantic cache, which is disposable and clears itself on
a mismatch. These rows are the only record of which side effects have already run, so a
misread checkpoint could re-run one that already happened. Finish or abandon those threads
with the version that wrote them, then delete the file.

## Limits

- **`stream()` is not checkpointed.** Only `run()` and `resume()` are. A streamed run has
  no `thread_id` parameter.
- **Resume needs the same tools and the same agent shape.** The checkpoint stores the
  conversation, not your registry. Resuming with a tool renamed or removed will not end
  well — if that tool was the in-flight call, resume raises `UnresolvedToolCallError`
  rather than telling the model the side effect did not happen.
- **The at-least-once boundary is real.** If a tool has an externally-visible side effect,
  mark it `idempotent=False` and handle the resolution, or make the effect idempotent on
  the tool's own side. Leaving the default on a `charge_card` is a decision to
  double-charge on a crash.

## See also

- [StateGraph](graph.md) — the same guarantee at node granularity, for workflows that
  branch and loop.
- [Agent](agent.md) — the run loop these checkpoints are taken from.
