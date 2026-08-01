# Migrating from LangChain / LangGraph

This page maps LangChain/LangGraph constructs to their `actants`
equivalents. The two frameworks have different scopes — LangChain has
many provider and storage integrations that `actants` does not — so a
migration is most straightforward for code that uses the agent loop and
tool calling.

## API mapping

| LangChain | actants |
|---|---|
| `ChatOpenAI(...)` | `LLM(provider="openai", model="...")` |
| `ChatOllama(...)` | `LLM()` (default) or `LLM(model="...")` |
| `@tool` (decorator) | `ToolRegistry.register_function(name, desc, fn, input_schema=...)` |
| `create_react_agent(llm, tools)` | `Agent(llm=llm, tools=registry)` |
| `agent.invoke({"messages": [...]})` | `await agent.run("...")` |
| `agent.stream(...)` | `agent.stream("...")` (typed events instead of dicts) |
| `MessagesPlaceholder` + `prompt` | `ConversationMemory(system="...")` |
| `MultiServerMCPClient(...)` | `MCPClient({...})` |
| `LangSmith` tracing | OpenTelemetry GenAI spans (any OTLP backend) |

## Example — a tool-calling agent

LangGraph:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


agent = create_react_agent(ChatOpenAI(model="gpt-4o"), [get_weather])
result = agent.invoke({"messages": [("user", "weather in Paris?")]})
```

actants:

```python
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
result = await agent.run("weather in Paris?")
```

## Differences worth knowing before you port

- `actants` is async-only; LangChain offers both sync and async paths.
- `actants` does not have LangChain's catalog of provider, vector store,
  document loader, and retriever integrations. If you depend on a specific
  one, you will need to re-implement it or keep that part on LangChain.
- `actants` exposes both an MCP client and an MCP server; LangChain's MCP
  story currently focuses on the client side.
- Tracing uses OpenTelemetry GenAI semantic conventions, which any OTLP
  collector can consume (Phoenix, Langfuse, Logfire, Datadog, etc.).
