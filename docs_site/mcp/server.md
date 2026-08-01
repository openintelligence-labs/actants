# MCP Server — expose your agent in two lines

Turn any agent into an MCP server. Other apps (Claude Desktop, IDEs, peer agents) can then call your agent's tools.

## Install

```bash
pip install 'actants[mcp]'
```

## Two lines

```python
from actants.mcp import serve

serve(agent)  # stdio (default)
serve(agent, transport="streamable-http", port=8000)  # HTTP
```

That's it.

## What gets exposed

Every tool registered on the agent's `ToolRegistry` becomes an MCP tool. Names, descriptions, JSON Schemas — all preserved verbatim.

## Use with Claude Desktop

In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-agent": {
      "command": "python",
      "args": ["/path/to/your_agent_server.py"]
    }
  }
}
```

Where `your_agent_server.py` ends with `serve(agent)`.

## Embedding in a larger app

If you need to mount the MCP server inside another web app, use `build_server()`:

```python
from actants.mcp import build_server

mcp_server = build_server(agent, name="my-agent")
# mcp_server is a FastMCP instance — mount via your ASGI framework
```

## Caveat: stdio servers must not print to stdout

stdio MCP uses stdout for the JSON-RPC stream. If your tools `print(...)` to stdout, they'll corrupt the protocol. Route logs to stderr (use `actants.observability.setup_logging` — it does this by default).
