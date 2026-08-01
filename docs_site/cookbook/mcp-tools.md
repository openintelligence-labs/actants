# Cookbook — Connect to an MCP server

Use `mcp-server-git` (a real MCP server) as agent tools.

```python
import asyncio
from actants import Agent, LLM
from actants.mcp import MCPClient


async def main():
    async with MCPClient(
        {
            "git": {"command": "uvx", "args": ["mcp-server-git", "--repository", "."]},
        }
    ) as mcp:
        print(f"Loaded {len(mcp.tools())} tools from MCP servers:")
        for tool in mcp.tools()[:5]:
            print(f"  - {tool.name}")

        agent = Agent(llm=LLM(model="llama3.2"), tools=mcp.tools())
        result = await agent.run(
            "What's the current branch and last commit message? "
            "Use the git__status and git__log tools."
        )
        print("\nAnswer:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

## Install

```bash
pip install 'actants[mcp]'
uv tool install mcp-server-git    # or: pip install mcp-server-git
ollama pull llama3.2
```

## What this demonstrates

- **Live MCP integration** — real subprocess, real protocol roundtrip
- **Auto tool discovery** — every tool the server exposes is registered automatically
- **Name prefixing** — `git__status`, `git__log` (server name as prefix prevents collisions)
