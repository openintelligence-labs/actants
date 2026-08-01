# Quickstart

## 1. Install

```bash
pip install actants
ollama serve                  # if not running
ollama pull llama3.2          # the model LLM() uses by default
```

`llama3.2` is the default model. To use one you have already pulled, pass it
explicitly (`LLM(model="qwen2.5:7b")`) or set `ACTANTS_MODEL`. If the model is
missing, actants lists the models you do have and the `ollama pull` command to
fix it.

## 2. First agent

```python
import asyncio
from actants import Agent


async def main():
    agent = Agent()  # Ollama default
    result = await agent.run("In one sentence: what is local-first software?")
    print(result.content)


asyncio.run(main())
```

No API key. No signup. Runs offline.

## 3. Add a tool

```python
import asyncio
from actants import Agent, LLM, ToolRegistry


async def main():
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
    print(result.content)  # "17 + 25 = 42."


asyncio.run(main())
```

The model decides when to call `add`, you don't have to wire it up.

## 4. Stream events as they happen

```python
from actants.agents import AgentTextDelta, AgentToolCallStarted, AgentRunCompleted

async for event in agent.stream("explain transformers"):
    match event:
        case AgentTextDelta(text=t):
            print(t, end="", flush=True)
        case AgentToolCallStarted(call=c):
            print(f"\n→ {c.name}({c.arguments})")
        case AgentRunCompleted(content=final):
            print(f"\n[done]")
```

## 5. Switch to a cloud provider when you need to

```python
from actants import Agent, LLM

agent = Agent(llm=LLM(provider="openai", model="gpt-4o"))  # needs OPENAI_API_KEY
agent = Agent(llm=LLM(provider="anthropic", model="claude-3-5-sonnet"))
agent = Agent(llm=LLM(provider="groq", model="llama-3.3-70b-versatile"))
```

Same `Agent`, same `tools`, different model. Your code doesn't change.

## Next steps

- [Concepts → Agent](concepts/agent.md) — what the Agent class does
- [MCP server](mcp/server.md) — expose your agent to other apps
- [Cookbook](cookbook/research-agent.md) — runnable recipes
