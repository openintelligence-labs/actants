# Tools

Tools are async functions the model can call. Register them with `ToolRegistry`.

## Register a function

<!-- docs-test: run -->

```python
from actants import ToolRegistry

tools = ToolRegistry()


async def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


tools.register_function("search", "Search the web for a query", search)
```

The input schema is derived from the handler's type annotations, so every
parameter must be annotated. Pass `input_schema=` explicitly when you need
something JSON Schema can express but annotations cannot — enums, ranges,
nested objects:

<!-- docs-test: run -->

```python
from actants import ToolRegistry

tools = ToolRegistry()


async def set_status(state: str) -> str:
    return state


tools.register_function(
    "set_status",
    "Set the run status",
    set_status,
    input_schema={
        "type": "object",
        "properties": {"state": {"type": "string", "enum": ["queued", "running", "done"]}},
        "required": ["state"],
    },
)
```

The `input_schema` is a JSON Schema; the model uses it to construct calls.

## Permission checks

```python
async def check(name: str, args: dict) -> bool:
    return name != "delete_database"  # block dangerous tools


tools = ToolRegistry(permission_check=check)
```

## Use them with an Agent

```python
from actants import Agent, LLM

agent = Agent(llm=LLM(), tools=tools)
await agent.run("search for python frameworks")
```

## Tools from MCP servers

```python
from actants.mcp import MCPClient

async with MCPClient({"git": {"command": "uvx", "args": ["mcp-server-git"]}}) as mcp:
    agent = Agent(llm=LLM(), tools=mcp.tools())
```

See [MCP client](../mcp/client.md).
