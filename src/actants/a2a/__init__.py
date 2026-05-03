"""Agent2Agent (A2A) Protocol integration for actants.

Two-line server::

    from actants.a2a import serve
    serve(agent, port=9000)   # mounts /.well-known/agent-card.json + JSON-RPC at /

Two-line client (call a remote agent as a tool)::

    from actants.a2a import RemoteAgent
    remote = RemoteAgent("https://other-agent.example.com")
    agent = Agent(llm=LLM(), tools=[remote])

Requires the official A2A SDK: ``pip install actants[a2a]``.
"""

from __future__ import annotations

from actants.a2a.card import build_agent_card
from actants.a2a.client import RemoteAgent
from actants.a2a.server import build_app, serve

__all__ = ["RemoteAgent", "build_agent_card", "build_app", "serve"]
