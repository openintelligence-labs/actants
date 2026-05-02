"""Agent2Agent (A2A) Protocol integration for agentic-kit.

Two-line server::

    from agentic_kit.a2a import serve
    serve(agent, port=9000)   # mounts /.well-known/agent-card.json + JSON-RPC at /

Two-line client (call a remote agent as a tool)::

    from agentic_kit.a2a import RemoteAgent
    remote = RemoteAgent("https://other-agent.example.com")
    agent = Agent(llm=LLM(), tools=[remote])

Requires the official A2A SDK: ``pip install agentic-kit[a2a]``.
"""

from __future__ import annotations

from agentic_kit.a2a.card import build_agent_card
from agentic_kit.a2a.client import RemoteAgent
from agentic_kit.a2a.server import build_app, serve

__all__ = ["RemoteAgent", "build_agent_card", "build_app", "serve"]
