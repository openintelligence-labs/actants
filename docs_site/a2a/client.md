# A2A Client — call remote agents as tools

`RemoteAgent` wraps a remote A2A endpoint as a Tool. Drop it into your local agent's tool registry; the local agent treats it like any other tool.

## Install

```bash
pip install 'actants[a2a]'
```

## Two lines

```python
from actants.a2a import RemoteAgent

remote = RemoteAgent("https://research-agent.example.com")
agent = Agent(llm=LLM(), tools=[remote])
```

The Agent Card is fetched lazily on the first call.

## Custom name and description

By default, `RemoteAgent` derives a name from the URL (`a2a__research_agent_example_com`). For nicer names:

```python
remote = RemoteAgent(
    "https://research-agent.example.com",
    name="research_expert",
    description="Send research questions to the research expert",
)
```

## Multiple remote peers

```python
from actants import ToolRegistry

tools = ToolRegistry()
tools.register(RemoteAgent("http://agent-a:9000", name="agent_a"))
tools.register(RemoteAgent("http://agent-b:9000", name="agent_b"))
agent = Agent(llm=LLM(), tools=tools)
```

The local model decides which remote to call based on each Agent Card's description.

## Caveats

- **Streaming is one-way.** The remote streams to the local agent over SSE; the local agent does not stream back.
- **No cancellation.** A2A supports `tasks/cancel` but actants's local agent loop is not interruptible mid-step yet — cancel becomes a `TASK_STATE_CANCELED` once the current step finishes.
- **Auth is delegated.** Pass `Authorization` etc. via the underlying httpx client; A2A spec defers to standard HTTP auth.
