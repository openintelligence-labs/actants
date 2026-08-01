# Cookbook — Two agents over A2A

Two terminals: one runs an agent as an A2A server, the other has a local agent that calls it as a tool.

## Terminal 1 — math-expert server

```python
# math_server.py
from actants import Agent, LLM, ToolRegistry
from actants.a2a import serve


def build_agent() -> Agent:
    tools = ToolRegistry()

    async def square(n: int) -> int:
        return n * n

    tools.register_function(
        "square",
        "Square an integer",
        square,
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    return Agent(llm=LLM(model="llama3.2"), tools=tools)


if __name__ == "__main__":
    serve(build_agent(), name="math-expert", port=9000)
```

```bash
pip install 'actants[a2a]'
python math_server.py
```

Visit `http://127.0.0.1:9000/.well-known/agent-card.json` to see the auto-generated card.

## Terminal 2 — caller

```python
# caller.py
import asyncio
from actants import Agent, LLM, ToolRegistry
from actants.a2a import RemoteAgent


async def main():
    tools = ToolRegistry()
    tools.register(RemoteAgent("http://127.0.0.1:9000", name="math_expert"))

    agent = Agent(llm=LLM(model="llama3.2"), tools=tools)
    result = await agent.run("Ask the math_expert what 9 squared is.")
    print(result.content)


asyncio.run(main())
```

```bash
python caller.py
```

## What this demonstrates

- **Two agents in two processes** — each independent, no shared memory
- **Standard A2A** — any A2A-compatible client (LangGraph, CrewAI, etc.) can call the math-expert
- **Tool composition** — the caller's agent treats the remote agent like any other tool
