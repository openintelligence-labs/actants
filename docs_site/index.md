# actants

A Python framework for building LLM agents. Defaults to Ollama for local
development; integrates OpenAI, Anthropic, Gemini, Groq, and Mistral via
opt-in extras. Includes MCP and A2A clients and servers, an embeddings
client, SQLite-based storage helpers, OpenTelemetry GenAI tracing, and a
CLI scaffold.

```python
import asyncio
from actants import Agent


async def main():
    agent = Agent()
    result = await agent.run("Say hello.")
    print(result.content)


asyncio.run(main())
```

## At a glance

| Property | Notes |
|---|---|
| Default LLM provider | Ollama (no API key required) |
| Cloud providers | OpenAI, Anthropic, Gemini, Groq, Mistral (opt-in extras) |
| Concurrency | async / await |
| MCP | client + server (`actants.mcp`, requires `[mcp]` extra) |
| A2A | client + server (`actants.a2a`, requires `[a2a]` extra) |
| Tracing | OpenTelemetry GenAI semantic conventions |
| Telemetry from the framework itself | none |
| License | MIT |
| Python | 3.12+ |

## Where to start

- **[Quickstart](quickstart.md)** — install, run, add a tool
- **[Installation](installation.md)** — extras and platform notes
- **[Configuration](configuration.md)** — env vars, settings, app paths
- **[Concepts → Agent](concepts/agent.md)** — `Agent`, memory, hooks, streaming
- **[MCP](mcp/server.md)** — expose or consume tools over the protocol
- **[A2A](a2a/server.md)** — expose or call peer agents
- **[Cookbook](cookbook/research-agent.md)** — runnable end-to-end recipes
- **[API reference](api/index.md)** — every public symbol
