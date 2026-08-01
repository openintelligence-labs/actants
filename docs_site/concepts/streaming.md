# Streaming

`Agent.stream()` yields lifecycle events as they happen — token-level text, tool dispatch, step boundaries, final answer.

## Event types

```python
from actants.agents import (
    AgentTextDelta,  # token chunk
    AgentToolCallStarted,  # tool dispatch beginning
    AgentToolCallCompleted,  # tool dispatch returned (ok or not)
    AgentStepCompleted,  # one LLM call + tool round done
    AgentRunCompleted,  # final answer (terminal)
)
```

## Pattern

```python
async for event in agent.stream("explain transformers"):
    match event:
        case AgentTextDelta(text=t):
            print(t, end="", flush=True)
        case AgentToolCallStarted(call=c):
            print(f"\n→ {c.name}({c.arguments})")
        case AgentToolCallCompleted(value=v, ok=True):
            print(f"  ✓ {v}")
        case AgentToolCallCompleted(value=err, ok=False):
            # When ok=False, value holds the error message
            print(f"  ✗ {err}")
        case AgentRunCompleted(content=final):
            print(f"\n[done — {len(final)} chars]")
```

After streaming completes, `agent.memory` is updated to match what `agent.run()` would have produced.

## When to use stream vs run

- `agent.stream()` — UI updates, log live, cancel mid-run, debugging
- `agent.run()` — batch processing, scripts, when you only need the final result
