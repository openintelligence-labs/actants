# actants

[![PyPI](https://img.shields.io/pypi/v/actants)](https://pypi.org/project/actants/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

A Python framework for building LLM agents. Defaults to Ollama for local
development; integrates OpenAI, Anthropic, Gemini, Groq, and Mistral via
opt-in extras. Includes MCP (Model Context Protocol) and A2A (Agent2Agent
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
| `mcp` | MCP client + server |
| `a2a` | A2A client + server |
| `cache` | sqlite-vec semantic cache |
| `cli` | Click + Rich CLI helpers |
| `all` | OpenAI + Anthropic + cache + cli |

```bash
pip install 'actants[openai,anthropic,mcp,a2a]'
```

For the default Ollama provider, also install
[Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
```

## Quickstart

```python
import asyncio
from actants import Agent

async def main():
    agent = Agent()                              # Ollama, llama3.2 by default
    result = await agent.run("Say hello.")
    print(result.content)

asyncio.run(main())
```

## Tools

Register async functions as tools and pass them to an `Agent`:

```python
from actants import Agent, LLM, ToolRegistry

tools = ToolRegistry()

async def add(a: int, b: int) -> int:
    return a + b

tools.register_function(
    "add",
    "Add two integers",
    add,
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)

agent = Agent(llm=LLM(model="llama3.2"), tools=tools)
result = await agent.run("What is 17 + 25?")
```

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
from actants import Agent, LLM

Agent(llm=LLM())                                                    # Ollama (default)
Agent(llm=LLM(provider="openai", model="gpt-4o"))                   # OPENAI_API_KEY
Agent(llm=LLM(provider="anthropic", model="claude-3-5-sonnet"))     # ANTHROPIC_API_KEY
Agent(llm=LLM(provider="groq", model="llama-3.3-70b-versatile"))    # GROQ_API_KEY
```

See [Configuration](https://github.com/openintelligence-labs/actants/blob/main/docs_site/configuration.md)
for the full list of environment variables.

## MCP

Expose an agent's tools over the Model Context Protocol:

```python
from actants.mcp import serve
serve(agent)                                              # stdio
serve(agent, transport="streamable-http", port=8000)      # HTTP
```

Consume tools from one or more MCP servers:

```python
from actants.mcp import MCPClient

async with MCPClient({
    "git": {"command": "uvx", "args": ["mcp-server-git"]},
    "fs":  {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
}) as mcp:
    agent = Agent(llm=LLM(), tools=mcp.tools())
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
from actants.a2a import RemoteAgent

remote = RemoteAgent("https://example.com")
agent = Agent(llm=LLM(), tools=[remote])
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
