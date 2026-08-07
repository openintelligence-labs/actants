# actants

[![PyPI](https://img.shields.io/pypi/v/actants)](https://pypi.org/project/actants/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

A Python framework for building LLM agents. Defaults to Ollama for local
development; integrates OpenAI, Anthropic, Gemini, and every major
OpenAI-compatible host (Groq, Mistral, xAI, DeepSeek, Together, Fireworks,
OpenRouter, Cerebras, Perplexity) via opt-in extras. Includes MCP (Model Context Protocol) and A2A (Agent2Agent
Protocol) clients and servers, an embeddings client, SQLite-based storage
helpers, OpenTelemetry GenAI tracing, and a Click + Rich CLI scaffold.

## Install

```bash
pip install actants
```

Optional extras:

| Extra | Adds |
|---|---|
| `openai` | OpenAI provider |
| `anthropic` | Anthropic provider |
| `gemini` | Google Gemini provider |
| `groq` | Groq provider |
| `mistral` | Mistral provider |
| `xai` | xAI / Grok provider |
| `deepseek` | DeepSeek provider |
| `together` | Together AI provider |
| `fireworks` | Fireworks AI provider |
| `openrouter` | OpenRouter provider |
| `cerebras` | Cerebras provider |
| `perplexity` | Perplexity provider |
| `mcp` | MCP client + server |
| `a2a` | A2A client + server |
| `cache` | sqlite-vec semantic cache |
| `cli` | Click + Rich CLI helpers |
| `all` | OpenAI + Anthropic + cache + cli |

```bash
pip install 'actants[openai,anthropic,mcp,a2a]'
```

For the default Ollama provider, also install
[Ollama](https://ollama.com), start it, and pull the default model:

```bash
ollama serve      # if it isn't already running
ollama pull llama3.2
```

`llama3.2` is what `LLM()` asks for unless you say otherwise. To use a model
you have already pulled, pass it explicitly — `LLM(model="qwen2.5:7b")` — or
set `ACTANTS_MODEL`.

## Quickstart

```python
import asyncio
from actants import Agent, LLM


async def main():
    agent = Agent(llm=LLM())  # Ollama, llama3.2 by default
    result = await agent.run("Say hello.")
    print(result.content)


asyncio.run(main())
```

If the model isn't on your Ollama server, actants tells you which models are
and what to run to fix it.

## Tools

Register async functions as tools and pass them to an `Agent`:

```python
from actants import Agent, LLM, ToolRegistry

tools = ToolRegistry()


async def add(a: int, b: int) -> int:
    return a + b


tools.register_function("add", "Add two integers", add)

agent = Agent(llm=LLM(model="llama3.2"), tools=tools)
result = await agent.run("What is 17 + 25?")
```

The JSON Schema the model sees is derived from `add`'s type annotations, so
every tool parameter must be annotated. Pass `input_schema=` explicitly for
anything annotations cannot express.

The model decides when to call the tool; `Agent` dispatches it and feeds
the result back through the tool-calling loop.

## Streaming

`Agent.stream()` yields typed events:

```python
from actants.agents import (
    AgentTextDelta,
    AgentToolCallStarted,
    AgentToolCallCompleted,
    AgentRunCompleted,
)

async for event in agent.stream("explain transformers in one paragraph"):
    match event:
        case AgentTextDelta(text=t):
            print(t, end="", flush=True)
        case AgentToolCallStarted(call=c):
            print(f"\n→ {c.name}({c.arguments})")
        case AgentToolCallCompleted(value=v):
            print(f"  ← {v}")
        case AgentRunCompleted():
            print()
```

## Durable execution

A run given a `checkpointer` and a `thread_id` persists its state after every LLM
completion and after each individual tool result, so a dead process can be
picked back up:

```python
from actants import Agent, LLM, SqliteCheckpointer, ToolRegistry

agent = Agent(
    llm=LLM(model="llama3.2"),
    tools=ToolRegistry(),
    checkpointer=SqliteCheckpointer("runs.db"),
    interrupt_before=["send_email"],
)

result = await agent.run("email the customer an apology", thread_id="job-7")
if result.interrupted:  # paused in front of send_email
    result = await agent.resume("job-7", approve=True)
```

The guarantee is narrow and stated rather than implied: resume is **at-most-once
for every tool call whose result was recorded**, and **at-least-once for the single
call that was in flight when the process died**. That one call is the irreducible
ambiguity — the process died before the tool could report — so actants surfaces it
rather than guessing. Tools registered `idempotent=False` are never auto-replayed;
they raise `UnresolvedToolCallError` and you resume with `resolve="retry"` or
`resolve="skip"`.

`interrupt_before` pauses in front of the named tools instead of dispatching them.
The pending call lives in the checkpoint, so the approval can come from another
process entirely.

Durability is opt-in per run: no `thread_id` means no storage is touched.
[Durability](https://actants.openintelligence-labs.org/concepts/durability/) has the
full contract, and
[StateGraph](https://actants.openintelligence-labs.org/concepts/graph/) applies the
same guarantee at node granularity for workflows that branch and loop.

## Record and replay

Wrap a provider to record a real run to JSONL, then replay it offline — no
network, no key, no server:

```python
from actants import Agent, LLM, OllamaProvider, ToolRegistry
from actants.testing import Recording, ReplayProvider, RunRecorder

recorder = RunRecorder("runs/booking.jsonl")
agent = Agent(llm=LLM(provider=recorder.wrap(OllamaProvider())), tools=ToolRegistry())
await agent.run("book a flight to Berlin")
recorder.close()

replayed = Agent(
    llm=LLM(provider=ReplayProvider(Recording.load("runs/booking.jsonl"))),
    tools=ToolRegistry(),
)
await replayed.run("book a flight to Berlin")  # identical, in milliseconds
```

Tool results are deliberately **not** replayed — the agent re-dispatches every call
against your real registry, so a bug in a tool's own logic cannot replay green. Point
tools at a fixture when replaying.

`EvalSuite` scores runs against cases and diffs two runs' cost, latency, and pass
rate, with trajectory scorers that catch what a final-answer check cannot — a refund
agent answering "done!" after calling `refund(cents=100000)` on a $10 order. See
[Testing agents](https://actants.openintelligence-labs.org/concepts/testing/).

## Switching providers

```python
from actants import Agent, LLM, LLMSettings

Agent(llm=LLM())  # Ollama (default)
Agent(llm=LLM(settings=LLMSettings(provider="openai", model="gpt-4o")))  # OPENAI_API_KEY
Agent(
    llm=LLM(settings=LLMSettings(provider="anthropic", model="claude-3-5-sonnet"))
)  # ANTHROPIC_API_KEY
Agent(
    llm=LLM(settings=LLMSettings(provider="groq", model="llama-3.3-70b-versatile"))
)  # GROQ_API_KEY
Agent(llm=LLM(provider="xai", model="grok-4"))  # XAI_API_KEY
Agent(llm=LLM(provider="deepseek", model="deepseek-chat"))  # DEEPSEEK_API_KEY
```

| Provider | API key env var | Notes | Verification |
|---|---|---|---|
| `ollama` | *(none)* | Default. Local, no key. | Live-verified |
| `openai` | `OPENAI_API_KEY` | | Unit-tested only |
| `anthropic` | `ANTHROPIC_API_KEY` | | Unit-tested only |
| `gemini` | `GEMINI_API_KEY` | | Unit-tested only |
| `groq` | `GROQ_API_KEY` | OpenAI-compatible | Unit-tested only |
| `mistral` | `MISTRAL_API_KEY` | OpenAI-compatible | Unit-tested only |
| `xai` | `XAI_API_KEY` | OpenAI-compatible | Unit-tested only |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible | Unit-tested only |
| `together` | `TOGETHER_API_KEY` | OpenAI-compatible | Unit-tested only |
| `fireworks` | `FIREWORKS_API_KEY` | OpenAI-compatible | Unit-tested only |
| `openrouter` | `OPENROUTER_API_KEY` | OpenAI-compatible | Unit-tested only |
| `cerebras` | `CEREBRAS_API_KEY` | OpenAI-compatible | Unit-tested only |
| `perplexity` | `PERPLEXITY_API_KEY` | OpenAI-compatible | Unit-tested only |

### What "verified" means here

actants supports 13 providers. That is a claim about code paths, not about how many
have been pointed at a live endpoint — so the table above says which is which, and
this section says exactly what was measured.

**Live-verified** means every one of these ran green against a real endpoint:
non-streaming completion, streaming (with usage reported at stream end and the
concatenated deltas matching the completed content), a tool call round-trip, a nested
structured-output extraction on the provider's *native* schema path, and a cost figure
that matches the provider's own reported token usage times the published price in
`actants.cost.PRICING`.

**Unit-tested only** means the provider is covered by the test suite against mocked
HTTP responses. Those mocks encode what actants *believes* the provider's wire format
is. That belief is derived from provider documentation and has not been confirmed
against the live API — so a provider marked this way may work perfectly, or may fail
on a detail the documentation did not describe. It is not a claim that it is broken;
it is a refusal to claim that it works.

Reproduce or extend the matrix — providers with no key present skip rather than fail,
so it is useful with a single key:

```bash
python -m verification.run                    # free providers only, no paid calls
python -m verification.run --yes              # every provider with a key present
python -m verification.run --only openai --yes
```

Paid APIs are never called without `--yes`, and the estimated spend is printed first.
See [`verification/`](verification/) for the harness and
[`docs/PROVIDER_VERIFICATION.md`](docs/PROVIDER_VERIFICATION.md) for the last recorded
run.

Cost tracking covers the models actants has verified prices for. A model with no
published price in `actants.cost.PRICING` is reported as *unknown*, not as `$0.00` —
`CostTracker.untracked_models` lists them, so a total that is really a lower bound
says so rather than looking like a free run.

Provider and model can also be set via `ACTANTS_PROVIDER` / `ACTANTS_MODEL`
environment variables, or by passing a provider instance as the first
positional argument to `LLM`. Since 0.5.3, the provider name alone also
works: `LLM(provider="openai", model="gpt-4o")`.

See [Configuration](https://github.com/openintelligence-labs/actants/blob/main/docs_site/configuration.md)
for the full list of environment variables.

## MCP

Expose an agent's tools over the Model Context Protocol:

```python
from actants.mcp import serve

serve(agent)  # stdio
serve(agent, transport="streamable-http", port=8000)  # HTTP
```

Consume tools from one or more MCP servers:

```python
from actants import Agent, LLM, ToolRegistry
from actants.mcp import MCPClient

async with MCPClient(
    {
        "git": {"command": "uvx", "args": ["mcp-server-git"]},
        "fs": {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
    }
) as mcp:
    registry = ToolRegistry()
    for tool in mcp.tools():
        registry.register(tool)
    agent = Agent(llm=LLM(), tools=registry)
```

The config shape matches Claude Desktop's `mcpServers`. Requires the
`[mcp]` extra and the official `mcp` Python SDK.

## A2A

Run an agent as an A2A server:

```python
from actants.a2a import serve

serve(agent, host="0.0.0.0", port=9000)
# /.well-known/agent-card.json + JSON-RPC at /
```

Call a remote A2A agent as a tool:

```python
from actants import Agent, LLM, ToolRegistry
from actants.a2a import RemoteAgent

registry = ToolRegistry()
registry.register(RemoteAgent("https://example.com"))
agent = Agent(llm=LLM(), tools=registry)
```

The Agent Card is auto-generated from the agent's tool registry. Streaming
uses Server-Sent Events. Requires the `[a2a]` extra and the official
`a2a-sdk` Python package.

## Tracing

`actants` emits OpenTelemetry GenAI semantic-convention spans
(`invoke_agent`, `chat`, `execute_tool`, `embeddings`). Cost is recorded
under `actants.cost.usd` because the OTel GenAI spec does not yet define a
cost attribute. Spans are forwarded to whichever OTLP collector you
configure; `actants` itself sends nothing.

## Benchmark

Measured on `actants` 1.0.0 installed from PyPI, against LangChain 1.3.14,
Pydantic AI 2.25.0, LlamaIndex 0.14.23, and the raw `ollama` client, on one
machine (Apple M4 Pro, Python 3.13.5, Ollama 0.32.6, `qwen2.5:7b`, 2026-08-06).
Framework overhead is isolated from model time with a recording proxy; latency
is p50 over 7 samples with framework order shuffled between rounds.

| | actants | LangChain | Pydantic AI | LlamaIndex | raw |
|---|---|---|---|---|---|
| Install (packages) | **18** | 38 | 98 | 63 | 12 |
| Install (site-packages) | **14.3 MB** | 36.0 MB | 106.7 MB | 126.6 MB | 11.3 MB |
| Cold import | **95.6 ms** | 343.9 ms | 724.1 ms | 539.4 ms | 105.9 ms |
| Overhead, completion | **7.54 ms** | 12.29 ms | 10.05 ms | 12.53 ms | 6.39 ms |
| Overhead, tool agent | **8.56 ms** | 18.45 ms | 13.47 ms | 934.21 ms | 7.74 ms |
| Overhead, structured | **7.71 ms** | 13.44 ms | 11.04 ms | 12.17 ms | 6.72 ms |
| LOC, three tasks | 24 | **23** | 32 | 25 | 49 |

`actants` has the smallest install and the lowest per-call overhead of the
frameworks tested — statistically tied with hand-written raw HTTP — and
**still loses the LOC row, by one line**. Schema inference in 1.0 cut tool
registration from 20 lines to 11, taking the total from 33 to 24 and moving
`actants` from last place to second; LangChain's `@tool` decorator still edges
it out, because `actants` requires an explicit `ToolRegistry` where LangChain
registers in place.

Model time dominates all wall-clock differences; these overheads are ~5 ms on
top of a ~150 ms model call. Single machine, small samples, no retrieval or
concurrency measured.

Full methodology, caveats, per-task snippets, and reproduction commands:
[docs/BENCHMARK.md](docs/BENCHMARK.md). Run it yourself with
`python benchmarks/run_benchmarks.py --runs 7`.

## Project layout

```
Agent           state, memory, hooks, streaming events
LLM             provider gateway, retry, fallback, cost, cache
Provider        Ollama, OpenAI, Anthropic, Gemini, Groq, Mistral
```

Opt-in modules: `mcp`, `a2a`, `embeddings`, `storage`, `cli`, `tracing`,
`observability`, `config`, `testing`.

## Stability

`actants` is **1.0**. Within the 1.x series, code using only the public API —
exactly what `actants.__all__` exports — keeps working and keeps meaning the
same thing. Names starting with `_` are private. The `mcp`, `a2a`, and `bench`
modules are provisional because the specs they track are still moving.

Deprecations get a `DeprecationWarning` plus at least two minor releases and
six months before removal, which never happens outside a major version. Run
`python -W error::DeprecationWarning -m pytest` against your suite to find out
whether an upgrade affects you before it does.

Full policy — what semver covers here, what is explicitly not promised, and how
it is enforced in CI: **[Stability policy](https://actants.openintelligence-labs.org/reference/stability/)**.

The package emits no telemetry.

## Links

- Issues: https://github.com/openintelligence-labs/actants/issues
- License: [MIT](LICENSE)
- Part of [Open Intelligence Labs](https://github.com/openintelligence-labs)
