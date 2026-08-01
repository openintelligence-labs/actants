# The ReAct loop

`Agent.run()` and `Agent.stream()` use the **ReAct** pattern (Reason → Act → Observe), with one critical update from the original 2022 paper: we use the model's native tool-calling, not `Thought:`/`Action:` prompt patterns.

## The loop

```
1. Send conversation messages → LLM
2. Did the model emit tool calls?
   ├─ yes: dispatch each tool
   │       append result as a `tool` message
   │       goto 1
   └─ no:  return the text answer (terminal)
3. Cap iterations at max_steps to prevent runaway loops
```

Three places implement this loop, all with the same algorithm:

| Where | What |
|---|---|
| `LLM.run_agent()` | One-shot, stateless |
| `Agent.run()` | Stateful with memory + hooks |
| `Agent.stream()` | Stateful, yields events live |

## Why we don't use `Thought:` prompts

The original ReAct paper (Yao et al., 2022) instructed models to emit `Thought:`, `Action:`, and `Observation:` lines as plain text, then parsed them with regex. Modern models (Llama 3+, Claude, GPT-4o, Gemini) emit structured tool calls directly via the function-calling API. The structured path is more reliable and simpler — no parsing, no escape-character bugs.

## What about other loop patterns?

- **Plan-and-Execute, Reflexion, Tree-of-Thought** — out of scope. Apps can build these on top of `LLM.complete()` themselves.
- **Code-as-action** — out of scope. It requires a sandboxed execution environment, which actants does not provide.
- **Multi-agent** — covered by [A2A](../a2a/server.md). One agent calls another over the standard protocol.

actants implements one loop. Other patterns can be built on top of `LLM.complete()`.
