# Migrating from Pydantic AI

Pydantic AI and `actants` have similar shapes — both are async, type-aware
agent frameworks with MCP support. The migration is mostly mechanical.

## API mapping

| Pydantic AI | actants |
|---|---|
| `Agent('openai:gpt-4o')` | `Agent(llm=LLM(provider="openai", model="gpt-4o"))` |
| `@agent.tool_plain` (infers schema from type hints) | `ToolRegistry.register_function(name, desc, fn, input_schema={...})` |
| `agent.run_sync(...)` | `asyncio.run(agent.run(...))` |
| `agent.run_stream(...)` | `agent.stream(...)` |
| `MCPServerStdio(...)` | `MCPClient({"name": {"command": ...}})` |
| `MCPServerStreamableHTTP(...)` | `MCPClient({"name": {"url": ...}})` |
| Logfire | OpenTelemetry GenAI spans (any OTLP backend) |

## Example

Pydantic AI:

```python
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o")


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


result = agent.run_sync("weather in Paris?")
print(result.output)
```

actants:

```python
import asyncio
from actants import Agent, LLM, ToolRegistry

tools = ToolRegistry()


async def get_weather(city: str) -> str:
    return f"sunny in {city}"


tools.register_function(
    "get_weather",
    "Get weather for a city",
    get_weather,
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

agent = Agent(llm=LLM(provider="openai", model="gpt-4o"), tools=tools)
result = asyncio.run(agent.run("weather in Paris?"))
print(result.content)
```

## Differences worth knowing before you port

- Tools require an explicit `input_schema`; Pydantic AI infers it from
  function annotations.
- `actants` is async-only — there is no `run_sync` equivalent.
- Tracing uses OpenTelemetry GenAI conventions, which any OTLP-compatible
  backend can consume.
- `actants` exposes both an MCP client and an MCP server, plus an A2A
  client and server.
