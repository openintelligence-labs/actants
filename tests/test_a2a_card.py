"""AgentCard generation tests."""

from __future__ import annotations

import pytest

a2a = pytest.importorskip("a2a")

from agentic_kit.a2a import build_agent_card  # noqa: E402
from agentic_kit.tools.registry import ToolRegistry  # noqa: E402


def test_card_has_required_fields():
    card = build_agent_card(
        name="my-agent",
        description="does things",
        version="1.0.0",
        url="http://localhost:9000",
    )
    assert card.name == "my-agent"
    assert card.description == "does things"
    assert card.version == "1.0.0"
    assert "text/plain" in card.default_input_modes
    assert "text/plain" in card.default_output_modes
    assert card.capabilities.streaming is True


def test_card_synthesizes_chat_skill_when_no_tools():
    card = build_agent_card(name="x", description="x", version="0.1.0", url="http://x", tools=None)
    assert len(card.skills) == 1
    assert card.skills[0].id == "chat"


def test_card_skills_one_per_tool():
    registry = ToolRegistry()

    async def search(query: str) -> str:
        return query

    async def fetch(url: str) -> str:
        return url

    registry.register_function(
        "search",
        "Search the web",
        search,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    registry.register_function(
        "fetch",
        "Fetch a URL",
        fetch,
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    card = build_agent_card(
        name="x", description="x", version="0.1.0", url="http://x", tools=registry
    )
    skill_ids = sorted(s.id for s in card.skills)
    assert skill_ids == ["fetch", "search"]
    fetch_skill = next(s for s in card.skills if s.id == "fetch")
    assert fetch_skill.description == "Fetch a URL"


def test_card_protocol_binding_jsonrpc():
    card = build_agent_card(name="x", description="x", version="0.1.0", url="http://x:9000")
    assert any(iface.protocol_binding == "JSONRPC" for iface in card.supported_interfaces)
