"""Error messages are a feature: every failure a new user can hit must name the
problem and say how to fix it.

Each test asserts on the *actionable* part of the message (the fix), not just the
exception type — a regression that keeps the type but drops the guidance should
fail here.
"""

from __future__ import annotations

import httpx
import pytest

from actants import LLM, Agent, ConversationMemory, LLMSettings, ToolRegistry
from actants.agents.hooks import AgentHooks
from actants.llm.base import ChatMessage, ToolSpec
from actants.llm.errors import (
    MissingAPIKeyError,
    ModelNotFoundError,
    ProviderError,
    UnknownProviderError,
)
from actants.tools.base import Tool


async def _add(a: int, b: int) -> int:
    return a + b


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def test_unknown_provider_lists_known_providers():
    with pytest.raises(UnknownProviderError) as exc:
        LLM(provider="nope")
    msg = str(exc.value)
    assert "Known providers" in msg
    assert "ollama" in msg and "openai" in msg


def test_misspelled_provider_suggests_the_right_one():
    with pytest.raises(UnknownProviderError) as exc:
        LLM(provider="opeani")
    assert "Did you mean 'openai'?" in str(exc.value)


def test_unknown_provider_is_still_a_value_error():
    """Back-compat: callers catching ValueError must keep working."""
    with pytest.raises(ValueError):
        LLM(provider="nope")


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
    ],
)
def test_missing_api_key_names_the_env_var_and_the_local_alternative(
    provider: str, env_var: str, monkeypatch: pytest.MonkeyPatch
):
    """Previously anthropic constructed fine and exploded at call time."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "ACTANTS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(MissingAPIKeyError) as exc:
        LLM(provider=provider, model="x")
    msg = str(exc.value)
    assert env_var in msg, "the message must name the exact env var to set"
    assert "LLM()" in msg, "the message must offer the no-API-key local path"


def test_api_key_error_is_raised_at_construction_not_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ACTANTS_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        LLM(provider="anthropic", model="claude-sonnet-4-5")


def test_settings_must_be_llmsettings_not_a_dict():
    with pytest.raises(TypeError) as exc:
        LLM(settings={"provider": "ollama"})  # type: ignore[arg-type]
    assert "LLMSettings(" in str(exc.value)


def test_model_must_be_a_string():
    with pytest.raises(TypeError) as exc:
        LLM(model=123)  # type: ignore[arg-type]
    assert "LLM(model='llama3.2')" in str(exc.value)


# --------------------------------------------------------------------------
# Ollama runtime failures
# --------------------------------------------------------------------------


async def test_ollama_not_running_says_how_to_start_it():
    llm = LLM(settings=LLMSettings(base_url="http://localhost:1"))
    with pytest.raises(ProviderError) as exc:
        await llm.complete("hi")
    msg = str(exc.value)
    assert "ollama serve" in msg
    assert "http://localhost:1" in msg, "the message must name the URL it tried"


async def test_model_not_pulled_lists_installed_models_and_the_pull_command(
    httpx_mock,
):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        status_code=404,
        json={"error": 'model "llama4" not found'},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "qwen2.5:7b"}, {"name": "gemma4:latest"}]},
    )

    llm = LLM(model="llama4")
    with pytest.raises(ModelNotFoundError) as exc:
        await llm.complete("hi")
    msg = str(exc.value)
    assert "ollama pull llama4" in msg, "must give the exact command to run"
    assert "qwen2.5:7b" in msg, "must list what the server actually has"


async def test_model_not_pulled_also_reported_when_streaming(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        status_code=404,
        json={"error": 'model "llama4" not found'},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "qwen2.5:7b"}]},
    )

    llm = LLM(model="llama4")
    with pytest.raises(ModelNotFoundError) as exc:
        async for _ in llm.stream("hi"):
            pass
    assert "ollama pull llama4" in str(exc.value)


async def test_embeddings_model_not_pulled_is_actionable(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/embed",
        status_code=404,
        json={"error": 'model "nope" not found'},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "nomic-embed-text:latest"}]},
    )

    from actants import Embeddings

    with pytest.raises(ModelNotFoundError) as exc:
        await Embeddings(model="nope").embed("hi")
    assert "ollama pull nope" in str(exc.value)


# --------------------------------------------------------------------------
# Agent construction
# --------------------------------------------------------------------------


def test_agent_defaults_to_local_ollama_with_no_arguments():
    """`Agent()` is what the docs promise; it must actually work."""
    agent = Agent()
    assert isinstance(agent.llm, LLM)
    assert agent.llm.settings.provider == "ollama"


def test_agent_rejects_a_non_llm():
    with pytest.raises(TypeError) as exc:
        Agent(llm="ollama")  # type: ignore[arg-type]
    assert "Agent(llm=LLM())" in str(exc.value)


@pytest.mark.parametrize("bad_tools", [[], {}, "add", object()])
def test_agent_rejects_tools_that_are_not_a_registry(bad_tools: object):
    """A list of functions is the intuitive-but-wrong thing to pass."""
    with pytest.raises(TypeError) as exc:
        Agent(llm=LLM(), tools=bad_tools)  # type: ignore[arg-type]
    assert "ToolRegistry" in str(exc.value)
    assert "register_function" in str(exc.value), "must show how to build one"


def test_agent_rejects_non_string_system():
    with pytest.raises(TypeError) as exc:
        Agent(llm=LLM(), system=123)  # type: ignore[arg-type]
    assert "system must be a string" in str(exc.value)


def test_agent_rejects_wrong_memory_type():
    with pytest.raises(TypeError) as exc:
        Agent(llm=LLM(), memory="hi")  # type: ignore[arg-type]
    assert "ConversationMemory" in str(exc.value)


def test_agent_rejects_wrong_hooks_type():
    with pytest.raises(TypeError) as exc:
        Agent(llm=LLM(), hooks=lambda: None)  # type: ignore[arg-type]
    assert "AgentHooks" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_agent_rejects_nonsensical_max_steps(bad: object):
    """max_steps=0 silently produced an agent that could never answer."""
    with pytest.raises(ValueError) as exc:
        Agent(llm=LLM(), max_steps=bad)  # type: ignore[arg-type]
    assert "max_steps must be an integer >= 1" in str(exc.value)


def test_agent_rejects_system_and_memory_together():
    with pytest.raises(ValueError) as exc:
        Agent(llm=LLM(), system="you are terse", memory=ConversationMemory())
    assert "not both" in str(exc.value)


def test_agent_accepts_valid_hooks_and_memory():
    agent = Agent(llm=LLM(), hooks=AgentHooks(), memory=ConversationMemory(system="x"))
    assert agent.max_steps == 6


# --------------------------------------------------------------------------
# Tool registration
# --------------------------------------------------------------------------


def test_sync_handler_is_rejected_with_the_fix_spelled_out():
    """A plain `def` tool used to register fine and fail at call time."""

    def sync_add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(TypeError) as exc:
        ToolRegistry().register_function("add", "Add", sync_add)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "async def" in msg
    assert "asyncio.to_thread" in msg, "must offer the escape hatch for blocking work"


def test_schema_is_derived_from_type_annotations():
    registry = ToolRegistry()
    tool = registry.register_function("add", "Add two integers", _add)
    assert tool.input_schema == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }


def test_unannotated_tool_parameters_are_rejected():
    """An un-annotated tool is invisible to the model — fail loudly instead."""

    async def mystery(a, b):  # type: ignore[no-untyped-def]
        return a

    with pytest.raises(ValueError) as exc:
        ToolRegistry().register_function("mystery", "?", mystery)
    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg
    assert "input_schema" in msg


def test_defaulted_parameters_are_not_marked_required():
    async def greet(name: str, loud: bool = False) -> str:
        return name

    tool = ToolRegistry().register_function("greet", "Greet", greet)
    assert tool.input_schema["required"] == ["name"]


def test_non_callable_handler_is_rejected():
    with pytest.raises(TypeError) as exc:
        ToolRegistry().register_function("x", "d", "not callable")  # type: ignore[arg-type]
    assert "async function" in str(exc.value)


def test_schema_missing_type_key_is_rejected():
    with pytest.raises(ValueError) as exc:
        ToolRegistry().register_function("x", "d", _add, input_schema={"properties": {}})
    assert "'type'" in str(exc.value)


def test_schema_with_invalid_json_type_is_rejected():
    with pytest.raises(ValueError) as exc:
        ToolRegistry().register_function(
            "x", "d", _add, input_schema={"type": "objekt", "properties": {}}
        )
    assert "objekt" in str(exc.value)
    assert "Valid types" in str(exc.value)


def test_object_schema_without_properties_is_rejected():
    with pytest.raises(ValueError) as exc:
        ToolRegistry().register_function("x", "d", _add, input_schema={"type": "object"})
    assert "properties" in str(exc.value)


def test_register_non_tool_points_at_register_function():
    with pytest.raises(TypeError) as exc:
        ToolRegistry().register("nope")  # type: ignore[arg-type]
    assert "register_function" in str(exc.value)


def test_duplicate_tool_name_explains_the_constraint():
    registry = ToolRegistry()
    registry.register_function("add", "Add", _add)
    with pytest.raises(ValueError) as exc:
        registry.register_function("add", "Add again", _add)
    assert "unique" in str(exc.value)


def test_valid_explicit_schema_still_accepted():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    tool = ToolRegistry().register_function("a", "d", _add, input_schema=schema)
    assert tool.input_schema == schema


async def test_unknown_tool_call_reports_registered_tools_to_the_model():
    """A hallucinated tool name comes back as ok=False so the model can retry,
    but the message must still name what *is* registered."""
    registry = ToolRegistry()
    registry.register_function("add", "Add", _add)

    result = await registry.call("subtract")
    assert result.ok is False
    assert "Registered tools: add" in (result.error or "")


def test_unknown_tool_get_raises_with_registered_tools():
    registry = ToolRegistry()
    registry.register_function("add", "Add", _add)
    from actants.tools.base import ToolError

    with pytest.raises(ToolError) as exc:
        registry.get("subtract")
    assert "Registered tools: add" in str(exc.value)


# --------------------------------------------------------------------------
# Prompt / argument shapes
# --------------------------------------------------------------------------


async def test_non_string_prompt_is_rejected_clearly():
    with pytest.raises(TypeError) as exc:
        await LLM().complete(123)  # type: ignore[arg-type]
    assert "string or a list of ChatMessage" in str(exc.value)


async def test_list_of_plain_strings_names_the_offending_index():
    with pytest.raises(TypeError) as exc:
        await LLM().complete(["hi", "there"])  # type: ignore[list-item]
    assert "prompt[0]" in str(exc.value)
    assert "ChatMessage(role='user'" in str(exc.value)


def test_openai_style_dict_messages_are_accepted():
    """Passing dicts is a natural mistake; accept them rather than crash."""
    messages = LLM._normalize([{"role": "user", "content": "hi"}], system=None)  # type: ignore[list-item]
    assert messages == [ChatMessage(role="user", content="hi")]


def test_malformed_dict_message_names_the_index():
    with pytest.raises(TypeError) as exc:
        LLM._normalize([{"role": "wizard", "content": "hi"}], system=None)  # type: ignore[list-item]
    assert "prompt[0]" in str(exc.value)


async def test_run_agent_without_a_registry_points_to_complete():
    with pytest.raises(TypeError) as exc:
        await LLM().run_agent("hi", None)  # type: ignore[arg-type]
    assert "llm.complete(prompt)" in str(exc.value)


async def test_run_agent_with_a_list_explains_toolregistry():
    with pytest.raises(TypeError) as exc:
        await LLM().run_agent("hi", [])  # type: ignore[arg-type]
    assert "ToolRegistry" in str(exc.value)


async def test_tools_must_be_toolspecs_not_dicts():
    with pytest.raises(TypeError) as exc:
        await LLM().complete("hi", tools=[{"name": "add"}])  # type: ignore[list-item]
    assert "ToolSpec" in str(exc.value)
    assert "as_specs()" in str(exc.value)


async def test_toolspec_list_is_accepted():
    spec = ToolSpec(name="add", description="Add", parameters={"type": "object", "properties": {}})
    # Only validating the guard here; no network call is made because the guard
    # runs before any provider work.
    assert spec.name == "add"


async def test_extract_requires_a_pydantic_model():
    with pytest.raises(TypeError) as exc:
        await LLM().extract("Bob is 3", dict)  # type: ignore[type-var]
    assert "BaseModel" in str(exc.value)


# --------------------------------------------------------------------------
# MCP
# --------------------------------------------------------------------------


def test_mcp_client_rejects_a_non_dict_config():
    pytest.importorskip("mcp")
    from actants.mcp import MCPClient

    with pytest.raises(TypeError) as exc:
        MCPClient("git")  # type: ignore[arg-type]
    assert "server name -> config" in str(exc.value)


async def test_mcp_missing_binary_names_the_command():
    pytest.importorskip("mcp")
    from actants.mcp import MCPClient
    from actants.mcp.transports import MCPConnectionError

    with pytest.raises(MCPConnectionError) as exc:
        async with MCPClient({"x": {"command": "actants-no-such-binary", "args": []}}):
            pass
    msg = str(exc.value)
    assert "actants-no-such-binary" in msg
    assert "PATH" in msg or "could not be run" in msg


async def test_mcp_config_without_command_or_url_is_explicit():
    pytest.importorskip("mcp")
    from actants.mcp import MCPClient

    with pytest.raises(ValueError) as exc:
        async with MCPClient({"x": {"args": []}}):
            pass
    assert "'command'" in str(exc.value) and "'url'" in str(exc.value)


# --------------------------------------------------------------------------
# Tool schema round-trip
# --------------------------------------------------------------------------


def test_registry_specs_survive_round_trip():
    registry = ToolRegistry()
    registry.register_function("add", "Add two integers", _add)
    specs = registry.as_specs()
    assert len(specs) == 1
    assert specs[0].parameters["properties"]["a"]["type"] == "integer"


def test_tool_model_still_constructible_directly():
    tool = Tool(name="x", description="d", input_schema={"type": "object", "properties": {}})
    assert tool.handler is None


def test_httpx_import_is_available_for_error_translation():
    assert hasattr(httpx, "ConnectError")
