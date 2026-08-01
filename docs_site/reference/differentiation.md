# Design notes

This page records the design choices behind `actants` and the trade-offs
they imply. It is not a comparison; the goal is to make the framework's
behavior predictable for users.

## Defaults

- **Default LLM provider** is Ollama. `Agent()` with no arguments expects
  an Ollama server at `http://localhost:11434`. Cloud providers are
  selected via `LLM(provider=...)` and require their own API keys.
- **Default embedding provider** is Ollama with `nomic-embed-text`.
- **Default storage** for local persistence helpers is SQLite, with
  `sqlite-vec` available under the `[cache]` extra for vector search.

These defaults exist so a fresh install runs without any cloud accounts.
Switching is one constructor argument.

## Async only

Every public I/O method is async. Apps that need a sync entrypoint can
wrap one call:

```python
import asyncio

asyncio.run(agent.run("..."))
```

There is no `run_sync` convenience; we keep one execution model so the
implementation does not need to maintain parallel sync and async paths.

## Telemetry

The framework itself does not make outbound network calls during import or
on user actions other than the LLM/tool/protocol calls that the user
explicitly initiated. There is no analytics, no anonymous usage reporter,
and no opt-out toggle (because there is nothing to opt out of). If you
configure OpenTelemetry, spans go to whichever collector you set up.

## Tool-calling loop

`Agent.run()` and `Agent.stream()` use each provider's native
function/tool-calling API rather than parsing a `Thought:` / `Action:` text
template. The model emits structured tool calls; the framework dispatches
them, appends results to the message history, and continues until the
model returns no further tool calls or `max_steps` is reached.

## Multi-agent composition

Multi-agent coordination uses the A2A protocol rather than a built-in
"crew" or "team" abstraction. Each agent runs as its own process and
exposes itself via A2A; orchestrating agents call peer agents through
`RemoteAgent`, which appears as an ordinary tool. This keeps the wire
format explicit and lets non-`actants` clients call your agents.

## Out of scope

The following are intentionally not part of the framework:

- Vector database integrations beyond SQLite
- Hosted SaaS or managed runtime
- Visual graph editors
- A "RAG" abstraction — embeddings and storage are exposed as primitives
- Code-execution sandboxes
- A synchronous API surface
