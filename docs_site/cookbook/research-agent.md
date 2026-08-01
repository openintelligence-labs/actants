# Cookbook — Local research agent

A complete agent that searches the web, reads pages, and writes a summary. Runs locally on Ollama, no API keys.

```python
import asyncio
import httpx
from selectolax.parser import HTMLParser

from actants import Agent, LLM, ToolRegistry
from actants.agents import ConversationMemory


async def web_search(query: str) -> str:
    """Search the web (DuckDuckGo)."""
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
    tree = HTMLParser(resp.text)
    results = []
    for link in tree.css("a.result__a")[:5]:
        title = link.text(strip=True)
        href = link.attributes.get("href", "")
        results.append(f"- {title}\n  {href}")
    return "\n".join(results) or "(no results)"


async def fetch_url(url: str) -> str:
    """Fetch a URL and return its visible text."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        resp = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
    tree = HTMLParser(resp.text)
    for tag in tree.css("script, style, nav, header, footer"):
        tag.decompose()
    return tree.body.text(separator=" ", strip=True)[:3000]


async def main(question: str) -> None:
    tools = ToolRegistry()
    tools.register_function(
        "web_search",
        "Search the web for a query",
        web_search,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    tools.register_function(
        "fetch_url",
        "Fetch a URL and return its text",
        fetch_url,
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    agent = Agent(
        llm=LLM(model="llama3.2"),
        tools=tools,
        memory=ConversationMemory(
            system=(
                "You are a research assistant. Use web_search to find sources, "
                "fetch_url to read them, then write a short cited answer."
            )
        ),
        max_steps=10,
    )
    result = await agent.run(question)
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main("What is the difference between MCP and A2A?"))
```

## Install

```bash
pip install actants httpx selectolax
ollama pull llama3.2
```

## What this demonstrates

- **Two tools, one agent** — the model decides when to search and when to fetch
- **Conversation memory** — multi-step workflow over many tool calls
- **Local-first** — entire pipeline runs offline (after the network requests for the searches themselves)
- **No API keys** — `ollama serve` is the only requirement

## Where to take it next

- Wrap as an MCP server: `serve(agent)` (see [MCP server](../mcp/server.md))
- Expose over A2A: `a2a_serve(agent)` (see [A2A server](../a2a/server.md))
- Add structured output: `await llm.extract(prompt, MyPydanticModel)`
