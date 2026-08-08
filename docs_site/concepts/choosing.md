# Agent or StateGraph?

actants gives you two ways to run an LLM over more than one step, and picking between
them is the first real decision you make. The question is not which is more capable. It
is **who decides what happens next.**

- With [`Agent`](agent.md), the **model** decides. You hand it tools and a goal; it picks
  a tool, sees the result, picks again, and stops when it judges the work done. You do
  not know in advance how many steps that takes, or in what order.
- With [`StateGraph`](graph.md), **your code** decides. You write the stages and the
  edges between them. The model is called *inside* a stage to do one bounded job — turn
  a question into search queries, extract the claims from this page, write this section
  — and ordinary Python decides what runs next.

A pipeline is not the consolation prize you accept when the agent loop won't fit. It is
usually the better design: predictable, cheap, debuggable one stage at a time, and
incapable of wandering somewhere you did not plan for. The honest rule is **if you can
write the steps down, write them down.** Reach for the agent loop when you genuinely
cannot.

## Which one

| | `Agent` | `StateGraph` |
|---|---|---|
| Decides the next step | the model | your code |
| Steps known before the run | no | yes |
| Cost of a run | varies with the model's choices | roughly fixed |
| A failure is localised to | a turn in a loop | a named node |
| Changing the sequence | rewrite the prompt, hope | edit an edge |
| Unit of durability | one tool call | one node |
| Add a step | describe a new tool | add a node and an edge |

Signals you want an `Agent`: the user's request is open-ended, the useful next tool
depends on what the last one returned, and you cannot enumerate the branches without
enumerating the inputs. A debugging assistant is the honest case — which file to read
next is only knowable after reading the last one.

Signals you want a `StateGraph`: you can describe the job as a sequence on a whiteboard,
the same shape runs every time, and the model's role in each stage is "transform this
input into that output". Most extraction, summarisation, research, and report-writing
work is this, even when it is *sold* as an agent.

## Two words called "agent"

"Agent" in product copy means software that goes off and does a job for you. "Agent" in
the technical sense means a tool-calling loop where the model chooses the next call. They
are not the same thing, and conflating them pushes people into the loop when a pipeline
would serve them better.

DeepDive, a research product in this same ecosystem, is marketed as a research agent and
contains zero `Agent` instances. Its work is a fixed sequence — question → generate
queries → search → scrape → extract claims → cross-reference → report — written as
ordinary Python, with `llm.extract()` and `llm.complete()` called inside each stage. It
adopted `StateGraph` and `SqliteCheckpointer` in a single PR precisely because it was
already a state machine; it just wasn't a persisted one. Nothing about the product's
behaviour changed. It became resumable.

## The same job, both ways

Summarise a document: read it, then write the summary.

As an agent, you register the reading as a tool and let the model decide when to call it:

<!-- docs-test: skip -->

```python
from actants import Agent, LLM, ToolRegistry

tools = ToolRegistry()


async def read_document(name: str) -> str:
    return open(name).read()


tools.register_function("read_document", "Read a document by name", read_document)

agent = Agent(llm=LLM(model="llama3.2"), tools=tools)
result = await agent.run("Summarise report.txt in three sentences.")
```

That is two steps if the model behaves and five if it decides to re-read the file, and
the only place to correct it is the prompt.

As a graph, you write the two steps down:

<!-- docs-test: run -->

```python
from typing import Any

from pydantic import BaseModel

from actants import END, LLM, StateGraph
from actants.testing import FakeLLMProvider, fake_completion


class State(BaseModel):
    name: str
    text: str = ""
    summary: str = ""


llm = LLM(
    provider=FakeLLMProvider([fake_completion("Three sentences about the report.")]),
    model="fake",
    tracing=False,
)


async def load(state: State) -> dict[str, Any]:
    return {"text": f"the contents of {state.name}"}


async def summarise(state: State) -> dict[str, Any]:
    result = await llm.complete(f"Summarise in three sentences:\n\n{state.text}")
    return {"summary": result.content}


graph: StateGraph[State] = StateGraph(State)
graph.add_node("load", load)
graph.add_node("summarise", summarise)
graph.set_entry_point("load")
graph.add_edge("load", "summarise")
graph.add_edge("summarise", END)

result = await graph.compile().invoke(State(name="report.txt"))
assert result.executed == ["load", "summarise"]
assert result.state.summary == "Three sentences about the report."
```

Two LLM calls become one. The read is a plain function, not something the model has to be
persuaded to call. When the summary comes out wrong, the node that produced it is named
in `result.executed` and can be run on its own.

## Durability reaches both

Resumability is not a reason to choose one shape over the other. Both get the same
guarantee through a different door:

<!-- docs-test: skip -->

```python
from actants import Agent, LLM, SqliteCheckpointer, ToolRegistry

store = SqliteCheckpointer("runs.db")

agent = Agent(llm=LLM(model="llama3.2"), tools=ToolRegistry(), checkpointer=store)
await agent.run("book the flight", thread_id="job-1")

compiled = graph.compile(checkpointer=store)
await compiled.invoke(State(name="report.txt"), thread_id="job-2")
```

One store holds both kinds of thread; they are tagged, so resuming one with the other's
`resume()` fails loudly rather than misreading the payload. What differs is the unit of
work that gets recorded: a tool call for the agent, a node for the graph. See
[Durability](durability.md) for the agent's contract and
[StateGraph → Durability](graph.md#durability) for the graph's.

## Mixing them

The two are not exclusive. [`agent_node`](graph.md#an-agent-as-a-node) drops a whole
`Agent` into one node of a graph, which is the shape to reach for when *one* stage of an
otherwise fixed pipeline is genuinely open-ended — a research pipeline whose gathering
step is a loop, sitting between a deterministic query builder and a deterministic report
writer. You get code-driven structure everywhere it is knowable and a model-driven loop
only where it isn't.

## If you are still unsure

Start with a `StateGraph`. Writing the stages out forces you to say what the job actually
is, and if a stage turns out to be genuinely unbounded you can swap it for an `agent_node`
without touching the rest. Going the other way — decomposing a working agent loop into
stages after the fact — means reconstructing a sequence that was never written down.

Next: [Agent](agent.md) for the loop, [StateGraph](graph.md) for the graph.
