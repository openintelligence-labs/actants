# MCP Client — consume MCP servers as tools

actants can connect to any MCP server (stdio or HTTP) and expose its tools to your agent. Three lines.

## Install

```bash
pip install 'actants[mcp]'
```

## Connect to one server

```python
from actants import Agent, LLM
from actants.mcp import MCPClient

async with MCPClient(
    {
        "git": {"command": "uvx", "args": ["mcp-server-git", "--repository", "."]},
    }
) as mcp:
    agent = Agent(llm=LLM(), tools=mcp.tools())
    await agent.run("show git status")
```

## Connect to multiple servers

```python
async with MCPClient(
    {
        "git": {"command": "uvx", "args": ["mcp-server-git"]},
        "fs": {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
        "co": {"url": "https://internal.co/mcp", "headers": {"Authorization": "Bearer ..."}},
    }
) as mcp:
    agent = Agent(llm=LLM(), tools=mcp.tools())
```

Tools from each server are name-prefixed (`git__status`, `fs__read_file`) so they don't collide.

## Config shape

actants uses **the same shape as Claude Desktop's `mcpServers` config**, so you can paste your existing config:

```python
{
    "name": {  # required: a unique key per server
        "command": "...",  # stdio: subprocess command
        "args": [...],  # stdio: args
        "env": {...},  # stdio: env vars (optional)
        # OR
        "url": "https://...",  # HTTP: server URL
        "headers": {...},  # HTTP: headers (optional)
    }
}
```

## Lower-level access

```python
async with MCPClient({"git": {...}}) as mcp:
    git = mcp.toolset("git")  # access one server's tools
    await git.session.list_tools()  # raw MCP session
```
