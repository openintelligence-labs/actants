# Testing agents

An agent's behaviour is a function of model, prompt, and tools. Change one and the only
honest way to know what broke is to run the thing again — which costs money, needs a
network, and is not reproducible.

`actants.testing` answers that in two halves:

- **Record and replay** turns one real run into an artifact you can re-run offline, in
  milliseconds, forever. It answers *did anything change?*
- **`EvalSuite`** scores runs against cases and diffs two runs' cost, latency, and pass
  rate. It answers *was the change good?*

Plus [`FakeLLMProvider`](../api/testing.md) for a unit test that never needed a real run
in the first place.

## Recording a run

`RunRecorder.wrap()` returns your provider with recording attached. Use it exactly as you
would the original — recording is observation, never interception:

```python
from actants import Agent, LLM, OllamaProvider, ToolRegistry
from actants.testing import RunRecorder

tools = ToolRegistry()

recorder = RunRecorder("runs/booking.jsonl", label="booking-v1")
agent = Agent(
    llm=LLM(provider=recorder.wrap(OllamaProvider()), model="qwen2.5:7b"),
    tools=tools,
)
result = await agent.run("book a flight to Berlin")
recorder.close()
```

What is captured is the **provider boundary** — the request that went on the wire and the
completion that came back. That is the one place every path through actants passes
through, and the only place a replay can serve an answer without a network.

`RunRecorder` is also a context manager, and `path=None` records to memory only, which is
what a test that wants no file on disk wants. `recorder.recording` is readable either way,
including mid-run.

The format is JSONL: one header line, then one line per LLM exchange. Diffable in a PR,
greppable, readable by anything that can read a file.

## Replaying it offline

```python
from actants import Agent, LLM, ToolRegistry
from actants.testing import Recording, ReplayProvider

recording = Recording.load("runs/booking.jsonl")
agent = Agent(
    llm=LLM(provider=ReplayProvider(recording), model="qwen2.5:7b"),
    tools=ToolRegistry(),
)
result = await agent.run("book a flight to Berlin")  # identical, offline
```

No network, no key, no server. Here is the whole loop, end to end and genuinely runnable:

<!-- docs-test: run -->

```python
from actants import Agent, LLM, ToolRegistry
from actants.testing import (
    FakeLLMProvider,
    ReplayProvider,
    RunRecorder,
    fake_completion,
    fake_tool_call_completion,
)

tools = ToolRegistry()


async def add(a: int, b: int) -> int:
    return a + b


tools.register_function("add", "Add two integers", add)

# Record. FakeLLMProvider stands in for the real one so this page's example needs
# nothing running; in your own code it would be OllamaProvider() or similar.
live = FakeLLMProvider(
    [
        fake_tool_call_completion("add", {"a": 2, "b": 3}),
        fake_completion("The answer is 5"),
    ]
)
recorder = RunRecorder()
recorded = await Agent(
    llm=LLM(provider=recorder.wrap(live), model="fake", tracing=False),
    tools=tools,
).run("what is 2+3?")

# Replay, offline.
replayed = await Agent(
    llm=LLM(provider=ReplayProvider(recorder.recording), model="fake", tracing=False),
    tools=tools,
).run("what is 2+3?")

assert replayed.content == recorded.content == "The answer is 5"
assert recorder.recording.tool_calls() == [("add", {"a": 2, "b": 3})]
```

## Tool results are not replayed

This is the design decision that matters most, so it is worth stating plainly.

**A replay serves the model's answers from the recording. It does not serve tool results.**
The agent re-dispatches every tool call against your real registry, and the tool's real
code runs.

That is deliberate. If tool results were replayed too, a change in a tool's own logic
could never be caught — a broken `calculate_total` would replay green forever, because the
recording would keep handing back the number it returned on the day it was correct.
Half the point of a regression suite for an agent is catching exactly that.

The consequence to plan for: **your tools run for real during a replay.** A replay of a
run that charges a card will charge the card again. Point tools at a fixture, a sandbox,
or a stub when replaying, in the same way you would for any other test that exercises
them.

`ReplayProvider` records what it saw, so a replayed run stays comparable: `.requests` is
what the run under test asked, `.served` is what it got, and `.exhausted` says whether the
whole recording was consumed. A diff of `.requests` against the recording is the answer to
"did my prompt change break anything".

## Matching modes

The choice says what the replay is *for*:

| `match=` | Behaviour | Use it for |
|---|---|---|
| `"sequence"` (default) | Serves exchange 0, then 1, then 2, whatever was asked. | Replaying against a **different prompt, model, or tool set**. The request will not match, and matching it is not the point. |
| `"request"` | Looks each request up by content; raises `RecordingMissError` on one that was never recorded. | A **determinism check**. It proves the run asked the same questions, not merely the same number of them. |

```python
from actants.testing import Recording, ReplayProvider

recording = Recording.load("runs/booking.jsonl")
strict = ReplayProvider(recording, match="request")
```

Running past the end of the recording raises rather than inventing an answer, and the
message says what it means: the run under test takes more steps than the recorded one
did, so a prompt or tool change made the model loop longer.

`reset()` rewinds a `ReplayProvider` so one can drive a second run.

A recording carries a format version in its header. A file this build cannot read fails
loudly on `load()` instead of being read with today's assumptions — a misread baseline
would report a passing replay of a run that never happened. For a long recording being
scanned for one thing, `iter_exchanges(path)` streams without holding the file in memory.

## EvalSuite

A suite is a set of named cases. Each case is a prompt (or, for a graph, a state) plus
the scorers a good answer has to satisfy. **All** of a case's scorers must pass for the
case to pass — a case is an assertion, not an average.

<!-- docs-test: run -->

```python
from actants import Agent, LLM, ToolRegistry
from actants.testing import (
    Contains,
    EvalCase,
    EvalSuite,
    FakeLLMProvider,
    ToolCalled,
    fake_completion,
    fake_tool_call_completion,
)

tools = ToolRegistry()


async def refund(order_id: str, cents: int) -> str:
    return f"refunded {cents} on {order_id}"


tools.register_function("refund", "Refund an order", refund, idempotent=False)

agent = Agent(
    llm=LLM(
        provider=FakeLLMProvider(
            [
                fake_tool_call_completion("refund", {"order_id": "A1", "cents": 1000}),
                fake_completion("Refunded order A1."),
            ]
        ),
        model="fake",
        tracing=False,
    ),
    tools=tools,
)

suite = EvalSuite(
    "refunds",
    [
        EvalCase(
            "a1",
            "refund order A1",
            scorers=[Contains("A1"), ToolCalled("refund", {"cents": 1000})],
            tags=("billing",),
        )
    ],
)

report = await suite.run(agent)
assert report.ok, report.summary()
```

The agent is reset between cases, so each case is an independent question rather than a
continuation of the previous one. `concurrency=` runs cases in parallel; the default of 1
is sequential, which is what a suite sharing one `Agent` needs.

## Scorers

| Scorer | Checks |
|---|---|
| `ExactMatch(expected)` | The final answer equals `expected`. `strip=True` and `case_sensitive=True` by default. |
| `Contains(*needles)` | The final answer contains all of them. Case-insensitive by default. |
| `ToolCalled(tool, arguments)` | The trajectory contains a call to `tool`, optionally with these arguments. |
| `ToolsCalledInOrder(*tools)` | The trajectory contains those tools as a **subsequence**. |
| `Predicate(fn, name=...)` | Whatever you write. Sync or async, returning `bool` or a `Score`. |

`ToolCalled` checks arguments as a **subset** by default, so a test pins the arguments it
cares about and stays green when a tool grows an optional one. Pass `exact=True` to
require the whole dict to match, and `times=n` to pin how many matching calls happened.

`ToolsCalledInOrder` is a subsequence and not an exact list on purpose: a model that
inserts an extra lookup between two required steps has still done the required steps in
the required order, and pinning the exact list produces a test that fails on every
harmless improvement.

A `Predicate` that *raises* is treated as a bug in the test, not a failing case — it
propagates as `ScorerError` rather than quietly turning red. A *run* that raises is the
opposite: it becomes a failing case with the exception recorded, so one broken case does
not abort the whole suite.

Any object with a `name` property and a `score(outcome)` method is a scorer; `Scorer` is a
runtime-checkable protocol.

## Why trajectory assertions

Checking the final answer is not enough, and the gap is not academic.

A refund agent that answers `"Done — your refund is on the way!"` while having called
`refund(cents=100000)` on a $10 order has produced a perfect final answer and a
catastrophe. `Contains("refund")` passes. `ExactMatch` passes. The only thing that catches
it is an assertion about *which tool ran with what arguments*:

```python
from actants.testing import ToolCalled, ToolsCalledInOrder

ToolCalled("refund", {"cents": 1000}, exact=True)
ToolsCalledInOrder("lookup_order", "verify_eligibility", "refund")
```

The second one catches a different failure the final answer also cannot see: a model that
skipped the eligibility check and refunded anyway. Its answer would be identical.

This is what `RunOutcome` carries for you — `tool_calls` in order, `tool_names` for a
quick order assertion, and `calls_of(name)` for every call to one tool. A failing
`ToolCalled` names the closest actual call, which is the difference between "assertion
failed" and "you passed the wrong city".

## Reading a report

`EvalReport` has `passed`, `failed`, `total`, `pass_rate`, and `ok` (true when every case
passed — what a CI job exits on). `summary()` is for a human and prints each failure with
its reason; `to_dict()` / `to_json()` are for a CI artifact, and include every case's
output, tool calls, and per-scorer verdicts.

<!-- docs-test: run -->

```python
from actants.testing import Contains, EvalCase, EvalSuite, RunOutcome, ToolCalled

# Scorers are plain objects — checkable directly, without running an agent.
outcome = RunOutcome(output="Refunded order A1.", cost_usd=0.002, latency_ms=310.0)
assert Contains("A1").score(outcome).passed
assert not ToolCalled("refund").score(outcome).passed

suite = EvalSuite("refunds", [EvalCase("a1", "refund A1", scorers=[Contains("A1")])])
assert len(suite) == 1
```

## Comparing two runs

`compare()` diffs a candidate report against a baseline — old model against new, before
against after:

```python
baseline = await suite.run(agent_on_the_big_model)
candidate = await suite.run(agent_on_the_small_model)

delta = candidate.compare(baseline)
print(delta.summary())
if delta.regressed:
    raise SystemExit("the cheaper model broke cases that used to pass")
```

`ReportDelta` gives `regressions` (passed before, fails now — the blocker list), `fixes`,
the cases present in only one report, and the cost/latency/pass-rate deltas. `self` is the
candidate, so a negative `cost_delta_usd` means the candidate is cheaper.

`cost_change_pct` is `None` rather than `0.0` or infinity when the baseline was free —
which it is for local Ollama. A report printing `+0%` for a run that went from free to $4
would be worse than one admitting the ratio is meaningless.

A cheaper model that costs 40% less and fails two more cases is a decision, not an
improvement. The point of the delta is to put both halves in front of you at once.

## Recording plus evals

The two compose into a suite that answers "did my prompt change break anything" without
touching a provider. Record the runs once, then point the eval at agents backed by
`ReplayProvider`. Remember what a replay does and does not cover: model answers come from
the recording, tool code runs for real, so this catches prompt and orchestration
regressions and tool-logic regressions — but not a change in how the *model* would have
responded to the new prompt. For that, re-record.

## See also

- [Testing API reference](../api/testing.md) — `FakeLLMProvider`, `fake_completion`, and
  the rest of the fakes.
- [Durability](durability.md) — the checkpointing these runs can be built on.
