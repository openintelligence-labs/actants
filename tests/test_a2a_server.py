"""A2A server tests using Starlette TestClient (no real server process)."""

from __future__ import annotations

import pytest

a2a = pytest.importorskip("a2a")
starlette = pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from agentic_kit.a2a import build_app  # noqa: E402
from agentic_kit.agents import Agent  # noqa: E402
from agentic_kit.llm.client import LLM  # noqa: E402
from agentic_kit.testing import FakeLLMProvider, fake_completion  # noqa: E402
from agentic_kit.tools.registry import ToolRegistry  # noqa: E402


def _build_agent() -> Agent:
    registry = ToolRegistry()

    async def echo(text: str) -> str:
        return text

    registry.register_function(
        "echo",
        "Echo text",
        echo,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    provider = FakeLLMProvider([fake_completion("Hello from the remote agent!")])
    return Agent(llm=LLM(provider=provider, model="fake"), tools=registry)


def test_well_known_agent_card_served():
    agent = _build_agent()
    app = build_app(
        agent,
        name="test-agent",
        description="for tests",
        version="1.0.0",
        base_url="http://testserver",
    )
    with TestClient(app) as client:
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "test-agent"
        assert body["version"] == "1.0.0"


def test_agent_card_lists_tool_skills():
    agent = _build_agent()
    app = build_app(agent, name="test-agent", base_url="http://testserver")
    with TestClient(app) as client:
        body = client.get("/.well-known/agent-card.json").json()
        skill_ids = sorted(s["id"] for s in body.get("skills", []))
        assert "echo" in skill_ids


def test_jsonrpc_endpoint_exists():
    """JSON-RPC endpoint should be mounted at root and respond to POSTs.

    We don't validate the full protocol here — just that the route is wired.
    """
    agent = _build_agent()
    app = build_app(agent, name="test-agent", base_url="http://testserver")
    with TestClient(app) as client:
        response = client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        # Either 200 with an RPC error envelope, or 4xx — just not 404.
        assert response.status_code != 404
