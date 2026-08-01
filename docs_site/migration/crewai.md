# Migrating from CrewAI

CrewAI organizes work as a `Crew` of `Agent`s executing `Task`s. `actants`
does not have a built-in crew abstraction; multi-agent coordination is
done by running each agent as its own process and exposing it over the
A2A protocol. Orchestrating agents call peer agents through `RemoteAgent`,
which appears in the local agent's tool registry.

## API mapping

| CrewAI | actants |
|---|---|
| `Agent(role=..., goal=..., tools=[...])` | `Agent(llm=LLM(), tools=registry)` plus a system prompt via `ConversationMemory` |
| `Task(description=..., agent=...)` | A `prompt` you pass to `agent.run()` |
| `Crew(agents=[a, b], tasks=[t1, t2])` | Two processes — a peer agent exposed via A2A, an orchestrator that calls it via `RemoteAgent` |
| `crew.kickoff()` | `await agent.run(...)` |
| Hierarchical / sequential process | Compose with [A2A](../a2a/server.md) calls |

## Example

CrewAI:

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="find facts", tools=[search])
writer = Agent(role="Writer", goal="write a summary", tools=[])
t1 = Task(description="Research transformers", agent=researcher, expected_output="notes")
t2 = Task(description="Write a 1-page summary using the notes", agent=writer)
Crew(agents=[researcher, writer], tasks=[t1, t2]).kickoff()
```

actants — the researcher runs as an A2A server, the writer orchestrates:

```python
# researcher_server.py
from actants import Agent, LLM, ToolRegistry
from actants.a2a import serve

tools = ToolRegistry()
# ... register search ...
serve(Agent(llm=LLM(), tools=tools), name="researcher", port=9000)
```

```python
# writer.py
from actants import Agent, LLM, ToolRegistry
from actants.a2a import RemoteAgent

tools = ToolRegistry()
tools.register(RemoteAgent("http://localhost:9000", name="researcher"))

writer = Agent(llm=LLM(), tools=tools)
await writer.run("Use the researcher to gather facts about transformers, then write a summary.")
```

## Differences worth knowing before you port

- There is no role/goal/backstory template; the system prompt is whatever
  you pass to `ConversationMemory(system=...)`.
- Multi-agent flow is explicit message passing over A2A rather than a
  built-in process model. The trade-off is more setup but simpler
  debugging — each agent has its own logs and standard A2A tooling works.
- `actants` is async-only; wrap with `asyncio.run` if you need a sync
  entrypoint.
- The framework itself does not emit telemetry.
