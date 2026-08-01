# Agent

`Agent` is the high-level class for running tool-calling LLM loops. It wraps `LLM` with conversation memory, lifecycle hooks, and streaming events.

## Minimum

```python
from actants import Agent

agent = Agent()  # Ollama, default model, no tools
result = await agent.run("hello")
print(result.content)
```

## With tools and a system prompt

```python
from actants import Agent, LLM, ToolRegistry
from actants.agents import ConversationMemory

tools = ToolRegistry()
# ... register tools ...

agent = Agent(
    llm=LLM(model="llama3.2"),
    tools=tools,
    memory=ConversationMemory(system="You are a concise math assistant."),
    max_steps=6,
)
```

## State across turns

`Agent` preserves conversation memory across calls to `run()` and `stream()`:

```python
await agent.run("My name is Alex.")
await agent.run("What's my name?")  # → "Your name is Alex."
agent.reset()  # clear conversation, keep system prompt
```

## Concurrent runs

Each `run()` and `stream()` works against a private copy of the conversation, seeded
from the agent's memory when the run starts, and commits its turn back atomically when
it finishes. Concurrent runs on one `Agent` therefore never see each other's partial
state — no run observes another's user message, and no run observes a half-written turn.

```python
import asyncio

agent = Agent(llm=LLM(model="llama3.2"))
alpha, beta, gamma = await asyncio.gather(
    agent.run("summarize A"),
    agent.run("summarize B"),
    agent.run("summarize C"),
)
```

Afterwards the agent's memory contains all three turns, each one contiguous. Commit
order is the order the runs finished.

A run that raises commits nothing, so a failed turn does not leave a user message
stranded without an answer. The same applies to a `stream()` the consumer abandons
part-way.

If you instead want later turns to build on earlier ones — a real multi-turn
conversation driven from several tasks — ask for serialization:

```python
agent = Agent(llm=LLM(model="llama3.2"), concurrency="serialized")
```

Runs then queue on a lock and execute one at a time, each seeing every turn committed
before it. This trades throughput for ordering.

Neither mode makes `Agent` safe across OS threads or processes; use one `Agent` per
event loop.

## Lifecycle hooks

```python
from actants.agents import AgentHooks


async def before(step, msgs):
    print(f"step {step}: {len(msgs)} messages in context")


async def after(step, completion):
    print(f"step {step}: {completion.usage.total_tokens} tokens")


async def on_tool(call, value):
    print(f"  → {call.name} = {value}")


async def on_error(exc):
    print(f"error: {exc}")


agent = Agent(
    llm=LLM(),
    hooks=AgentHooks(
        before_step=before,
        after_step=after,
        on_tool_call=on_tool,
        on_error=on_error,
    ),
)
```

## Streaming

Use `agent.stream()` to receive events as they happen. See [Streaming](streaming.md).
