# actants

[![PyPI](https://img.shields.io/pypi/v/actants)](https://pypi.org/project/actants/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

**A local-first AI agent framework. No API keys. No telemetry. Built for offline-first development.**

```python
from actants import Agent

agent = Agent()                                # Ollama default — no API key
print(await agent.run("Hello!"))               # Runs offline, on your laptop
```

That's the whole quickstart. No signup. No `OPENAI_API_KEY`. No phone-home.

---

## Why actants

| | What | Why it matters |
|---|---|---|
| **Local-first** | Defaults to Ollama. OpenAI is opt-in. | Privacy, cost, offline work — by default, not as a plugin. |
| **Zero telemetry** | No analytics. No phone-home. | What your agent sends, *you* sent. Nothing else. |
| **Lazy imports** | PEP 562 module-level lazy loading. | Only pay for what you use. |
| **Native MCP server + client** | A few lines to expose your agent. A few lines to consume one. | Your agent IS a Claude Desktop extension. |
| **Native A2A protocol** | Auto-generated Agent Card. Streaming SSE. | Agents in different frameworks talk to each other. |

---

## Install

```bash
pip install actants                                # core + Ollama
pip install 'actants[openai,anthropic]'            # cloud providers
pip install 'actants[mcp]'                         # MCP client + server
pip install 'actants[a2a]'                         # A2A client + server
pip install 'actants[cli]'                         # Click + Rich helpers
pip install 'actants[all]'                         # everything

ollama pull llama3.2                               # default local model
```

---

## What you can build

### A local agent with tools (zero config)

```python
import asyncio
from actants import Agent, LLM, ToolRegistry

async def main():
    tools = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    tools.register_function(
        "add", "Add two integers", add,
        input_schema={"type": "object",
                      "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"]},
    )

    agent = Agent(llm=LLM(model="llama3.2"), tools=tools)
    result = await agent.run("What is 17 + 25?")
    print(result.content)

asyncio.run(main())
```

### Stream every event

```python
from actants.agents import (
    AgentTextDelta, AgentToolCallStarted, AgentToolCallCompleted, AgentRunCompleted,
)

async for event in agent.stream("explain transformers in one paragraph"):
    match event:
        case AgentTextDelta(text=t):              print(t, end="", flush=True)
        case AgentToolCallStarted(call=c):        print(f"\n→ {c.name}({c.arguments})")
        case AgentToolCallCompleted(value=v):     print(f"  ✓ {v}")
        case AgentRunCompleted(content=final):    print(f"\n[done — {len(final)} chars]")
```

### Expose your agent as an MCP server

```python
from actants.mcp import serve
serve(agent)                                              # stdio (Claude Desktop)
serve(agent, transport="streamable-http", port=8000)      # HTTP for remote clients
```

Now Claude Desktop, IDEs, and any MCP-aware app can call your agent's tools.

### Consume any MCP server as agent tools

```python
from actants.mcp import MCPClient

async with MCPClient({
    "git": {"command": "uvx", "args": ["mcp-server-git"]},
    "fs":  {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
}) as mcp:
    agent = Agent(llm=LLM(), tools=mcp.tools())
    await agent.run("show me the git status of this repo")
```

### Speak A2A — your agent is callable from any A2A client

```python
from actants.a2a import serve
serve(agent, host="0.0.0.0", port=9000)
# Mounts /.well-known/agent-card.json + JSON-RPC at /
```

```python
from actants.a2a import RemoteAgent

remote = RemoteAgent("https://research-agent.example.com")
agent = Agent(llm=LLM(), tools=[remote])
await agent.run("Ask the research agent about transformers.")
```

### Switch providers without changing your code

```python
agent = Agent(llm=LLM(provider="openai", model="gpt-4o"))                 # OpenAI
agent = Agent(llm=LLM(provider="anthropic", model="claude-3-5-sonnet"))   # Claude
agent = Agent(llm=LLM(provider="groq", model="llama-3.3-70b-versatile"))  # Groq
agent = Agent()                                                           # Ollama (default)
```

OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama — same `LLM()` class, same `Agent`.

---

## Architecture

actants follows the **ReAct loop** (Reason → Act → Observe) using each model's native tool-calling — no `Thought:`/`Action:` prompt-engineering tricks. The model emits structured tool calls; we dispatch them and feed results back.

Three abstraction layers:

```
Agent           → state, memory, hooks, streaming events
LLM             → provider gateway, retry, fallback, cost, cache
BaseLLMProvider → Ollama / OpenAI / Anthropic / Gemini / Groq / Mistral
```

Plus opt-in modules: `mcp/`, `a2a/`, `embeddings/`, `storage/`, `cli/`, `tracing/`, `observability/`, `config/`, `testing/`.

---

## What we won't build

A framework's "no" list is more important than its "yes" list. We will not add:

- **Vector DB integrations** beyond SQLite (sqlite-vec scales well for local use)
- **Multi-agent metaphors** ("Crews", "Societies", "Workflows") — A2A covers it
- **RAG-as-a-feature** — embeddings + storage are primitives; RAG is an app pattern
- **Code-execution agents** — sandbox quality is a separate product
- **Visual graph builders**
- **Hosted SaaS / paid tier** — we sell nothing
- **Sync API** — async only, one way to do it

If you want any of those, grab a different framework.

---

## OpenTelemetry GenAI conformance

actants emits spans following [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

```
invoke_agent llama3.2          (CLIENT)
├── chat llama3.2              (CLIENT)
├── execute_tool search        (INTERNAL)
├── chat llama3.2              (CLIENT)
└── execute_tool fetch_url     (INTERNAL)
```

All `gen_ai.*` attribute names match the spec. Cost is namespaced under `actants.cost.usd` (the spec doesn't define a cost attribute). Opt into experimental attributes via `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`.

Works with Phoenix, Langfuse, Logfire, Datadog, any OTel-compatible backend.

---

## Status

- **License:** MIT
- **Python:** 3.12+

---

## Project & community

Part of [Open Intelligence Labs](https://github.com/openintelligence-labs) — a collection of independent local-first AI projects.

- **Issues / discussions:** GitHub Issues

If this framework saves you from writing another LLM wrapper, **star the repo**.
