"""RemoteAgent (A2A client) tests."""

from __future__ import annotations

import pytest

a2a = pytest.importorskip("a2a")

from actants.a2a import RemoteAgent  # noqa: E402
from actants.a2a.client import _slug_from_url  # noqa: E402
from actants.tools.base import Tool  # noqa: E402


def test_remote_agent_is_a_tool():
    remote = RemoteAgent("https://research.example.com")
    assert isinstance(remote, Tool)


def test_remote_agent_default_name_from_url():
    remote = RemoteAgent("https://research.example.com")
    assert remote.name == "a2a__research_example_com"


def test_remote_agent_custom_name():
    remote = RemoteAgent("https://x", name="my-research-agent")
    assert remote.name == "my-research-agent"


def test_remote_agent_input_schema_has_prompt():
    remote = RemoteAgent("https://x")
    assert remote.input_schema["properties"]["prompt"]["type"] == "string"
    assert "prompt" in remote.input_schema["required"]


def test_slug_handles_dashes_and_dots():
    assert _slug_from_url("https://my-cool-agent.co") == "a2a__my_cool_agent_co"


def test_slug_handles_missing_host():
    # Bare path / malformed URL
    out = _slug_from_url("not-a-url")
    assert out.startswith("a2a__")
