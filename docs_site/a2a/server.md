# A2A Server — expose your agent to peer agents

A2A (Agent2Agent) is the Linux Foundation protocol for inter-agent communication. actants speaks it natively.

## Install

```bash
pip install 'actants[a2a]'
```

## One line

```python
from actants.a2a import serve

serve(agent, host="0.0.0.0", port=9000)
```

This mounts:
- `GET /.well-known/agent-card.json` — agent discovery
- JSON-RPC at `/` — `message/send`, `message/stream`, `tasks/get`, etc.

## What the Agent Card looks like

Auto-generated from your agent's metadata and tool registry:

```json
{
  "name": "my-agent",
  "description": "An agent built with actants",
  "version": "0.1.0",
  "default_input_modes": ["text/plain"],
  "default_output_modes": ["text/plain"],
  "capabilities": {"streaming": true, "push_notifications": false},
  "supported_interfaces": [{"protocol_binding": "JSONRPC", "url": "http://..."}],
  "skills": [
    {"id": "search", "name": "search", "description": "Search the web", ...},
    {"id": "add", "name": "add", "description": "Add two integers", ...}
  ]
}
```

Each tool becomes a skill so peer agents can discover capabilities.

## Streaming

A2A streams responses over SSE. actants emits the standard task lifecycle —
`TASK_STATE_SUBMITTED` → `TASK_STATE_WORKING` → a final text artifact →
`TASK_STATE_COMPLETED` — and the final response is delivered as a single
`TaskArtifactUpdateEvent`. Token-level streaming over A2A is not currently supported.

## Mounting in an existing ASGI app

```python
from actants.a2a import build_app

a2a_app = build_app(agent, base_url="https://you.com")
# a2a_app is a Starlette app — mount or compose as needed
```

## A2A and MCP together

A single process can speak both. Use [`actants.mcp.serve`](../mcp/server.md) at one path and `actants.a2a.serve` at another. They don't conflict — MCP is vertical (your tools), A2A is horizontal (your agent talks to peers).
