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

| Provider | API key env var | Notes |
|---|---|---|
| `ollama` | *(none)* | Default. Local, no key. |
| `openai` | `OPENAI_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | |
| `gemini` | `GEMINI_API_KEY` | |
| `groq` | `GROQ_API_KEY` | OpenAI-compatible |
| `mistral` | `MISTRAL_API_KEY` | OpenAI-compatible |
| `xai` | `XAI_API_KEY` | OpenAI-compatible |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| `together` | `TOGETHER_API_KEY` | OpenAI-compatible |
| `fireworks` | `FIREWORKS_API_KEY` | OpenAI-compatible |
| `openrouter` | `OPENROUTER_API_KEY` | OpenAI-compatible |
| `cerebras` | `CEREBRAS_API_KEY` | OpenAI-compatible |
| `perplexity` | `PERPLEXITY_API_KEY` | OpenAI-compatible |

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

Measured against LangChain 1.3.14, Pydantic AI 2.21.0, LlamaIndex 0.14.23,
and the raw `ollama` client, on one machine (Apple M4 Pro, Python 3.13.5,
Ollama 0.32.4, `qwen2.5:7b`). Framework overhead is isolated from model time
with a recording proxy; latency is p50 over 7 samples with framework order
shuffled between rounds.

| | actants | LangChain | Pydantic AI | LlamaIndex | raw |
|---|---|---|---|---|---|
| Install (packages) | **18** | 38 | 98 | 63 | 12 |
| Install (site-packages) | **14.1 MB** | 35.9 MB | 105.7 MB | 126.7 MB | 11.3 MB |
| Cold import | **96.9 ms** | 343.2 ms | 742.5 ms | 565.7 ms | 116.8 ms |
| Overhead, completion | **6.03 ms** | 10.40 ms | 10.37 ms | 11.80 ms | 5.58 ms |
| Overhead, tool agent | **7.58 ms** | 17.33 ms | 12.88 ms | 951.32 ms | 7.62 ms |
| Overhead, structured | **6.21 ms** | 12.03 ms | 10.46 ms | 12.40 ms | 6.21 ms |
| LOC, three tasks | 33 | **23** | 32 | 25 | 49 |

`actants` has the smallest install and the lowest per-call overhead of the
frameworks tested — statistically tied with hand-written raw HTTP — and
**loses on tool ergonomics**: registering one tool takes 20 lines against
LangChain's 10, because released 0.5.3 requires a hand-written JSON Schema.
(Annotation-based schema inference has since landed on `main`; the table
measures the released package and will be regenerated when it ships.)

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

## Status

`actants` is pre-1.0. The public API listed in `actants.__all__` is
documented; everything else is implementation detail and may change. The
package emits no telemetry.

## Links

- Issues: https://github.com/openintelligence-labs/actants/issues
- License: [MIT](LICENSE)
- Part of [Open Intelligence Labs](https://github.com/openintelligence-labs)
