"""Provider-native constrained decoding for ``extract`` / ``extract_stream``.

The assertions that matter are on the **request body each provider actually receives**.
A schema that is merely built correctly but lands in the wrong place on the wire is a
400 from the provider, not a passing feature, so every native-path test here drives a
mocked transport and inspects what was sent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from actants.errors import UnsupportedSchemaError
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from actants.llm.client import LLM
from actants.llm.ollama import OllamaProvider
from actants.llm.structured import (
    ANTHROPIC_EXTRACT_TOOL,
    build_schema_plan,
    to_gemini_schema,
    to_strict_schema,
)


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    """A nested model, so every test exercises the $defs/$ref path."""

    name: str
    age: int
    address: Address
    nickname: str | None = None


PERSON_JSON = '{"name": "Ada", "age": 36, "address": {"street": "1 Main", "city": "London"}}'


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------


def test_strict_schema_inlines_defs_and_refs() -> None:
    """Nested models arrive as $ref into $defs; strict mode gets them inlined."""
    strict = to_strict_schema(Person.model_json_schema())
    assert "$defs" not in strict
    address = strict["properties"]["address"]
    assert "$ref" not in address
    assert address["properties"]["city"]["type"] == "string"


def test_strict_schema_requires_every_property_and_closes_objects() -> None:
    strict = to_strict_schema(Person.model_json_schema())
    for node in (strict, strict["properties"]["address"]):
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])


def test_strict_schema_makes_optional_fields_nullable_not_absent() -> None:
    """Strict mode cannot express "may be absent"; the encoding is a union with null."""
    strict = to_strict_schema(Person.model_json_schema())
    assert "nickname" in strict["required"]
    branches = strict["properties"]["nickname"]["anyOf"]
    assert {"type": "null"} in branches


def test_strict_schema_widens_a_defaulted_non_optional_field() -> None:
    class WithDefault(BaseModel):
        tags: list[str] = []

    widened: set[tuple[str, ...]] = set()
    strict = to_strict_schema(WithDefault.model_json_schema(), widened=widened)
    assert strict["required"] == ["tags"]
    assert strict["properties"]["tags"]["type"] == ["array", "null"]
    # The widening is forced by strict mode, so it is recorded rather than avoided —
    # that record is what lets extract read the resulting null back as "use the default".
    assert widened == {("tags",)}


def test_a_field_pydantic_already_made_nullable_is_not_a_defaulted_null() -> None:
    """``X | None`` accepts null as a real value, so it must not be stripped."""
    widened: set[tuple[str, ...]] = set()
    to_strict_schema(Person.model_json_schema(), widened=widened)
    assert ("nickname",) not in widened


def test_strict_schema_drops_only_the_keywords_strict_mode_rejects() -> None:
    """`pattern` and the numeric bounds are supported now; length bounds are not."""

    class Bounded(BaseModel):
        code: str = Field(pattern=r"^[A-Z]+$", min_length=2)
        age: int = Field(ge=0, le=120)

    props = to_strict_schema(Bounded.model_json_schema())["properties"]
    assert props["code"]["pattern"] == "^[A-Z]+$"
    assert "minLength" not in props["code"]
    assert props["age"]["minimum"] == 0 and props["age"]["maximum"] == 120


def test_strict_schema_keeps_the_field_description_on_an_annotated_ref() -> None:
    """pydantic v2 emits `$ref` with sibling keys; the description must survive inlining."""

    class Described(BaseModel):
        address: Address = Field(description="where they live")

    raw = Described.model_json_schema()
    assert raw["properties"]["address"]["$ref"]

    address = to_strict_schema(raw)["properties"]["address"]
    assert address["description"] == "where they live"
    assert set(address["properties"]) == {"street", "city"}


def test_strict_schema_unwraps_a_single_element_allof() -> None:
    """pydantic v1 (and some hand-written schemas) wrap an annotated $ref in `allOf`.

    Strict mode rejects `allOf`, but the wrapper carries no constraint — dropping it
    rather than unwrapping would discard the referenced model and leave a bare `{}`.
    """
    raw = {
        "type": "object",
        "properties": {"address": {"allOf": [{"$ref": "#/$defs/Address"}], "description": "where"}},
        "required": ["address"],
        "$defs": {Address.__name__: Address.model_json_schema()},
    }
    address = to_strict_schema(raw)["properties"]["address"]
    assert address["description"] == "where"
    assert set(address["properties"]) == {"street", "city"}


def test_strict_schema_rejects_a_multi_branch_allof() -> None:
    raw = {
        "type": "object",
        "properties": {"x": {"allOf": [{"type": "object"}, {"type": "string"}]}},
        "required": ["x"],
    }
    with pytest.raises(UnsupportedSchemaError, match="allOf"):
        to_strict_schema(raw)


def test_strict_schema_rejects_a_recursive_model() -> None:
    class Node(BaseModel):
        value: int
        child: Node | None = None

    with pytest.raises(UnsupportedSchemaError, match="recursive"):
        to_strict_schema(Node.model_json_schema())


def test_strict_schema_rejects_an_open_ended_dict() -> None:
    class Bag(BaseModel):
        payload: dict[str, Any]

    with pytest.raises(UnsupportedSchemaError, match="additionalProperties"):
        to_strict_schema(Bag.model_json_schema())


def test_gemini_schema_uses_openapi_types_and_nullable() -> None:
    schema = to_gemini_schema(Person.model_json_schema())
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["age"]["type"] == "INTEGER"
    assert schema["properties"]["address"]["properties"]["city"]["type"] == "STRING"
    assert schema["properties"]["nickname"]["nullable"] is True
    # Unlike strict mode, Gemini can express absence, so optional stays out of required.
    assert "nickname" not in schema["required"]


def test_gemini_schema_drops_formats_it_does_not_know() -> None:
    class Contact(BaseModel):
        email: str = Field(json_schema_extra={"format": "email"})
        when: str = Field(json_schema_extra={"format": "date-time"})

    props = to_gemini_schema(Contact.model_json_schema())["properties"]
    assert "format" not in props["email"]
    assert props["when"]["format"] == "date-time"


# ---------------------------------------------------------------------------
# Plan selection
# ---------------------------------------------------------------------------


def test_plan_declines_when_the_provider_has_no_native_mode() -> None:
    plan = build_schema_plan(Person, "none")
    assert plan.native is False
    assert plan.request_kwargs == {}
    assert plan.reason is not None


@pytest.mark.parametrize("mode", ["openai_json_schema", "anthropic_tool", "gemini", "ollama"])
def test_plan_takes_the_native_path_for_a_supported_schema(mode: str) -> None:
    plan = build_schema_plan(Person, mode)  # type: ignore[arg-type]
    assert plan.native is True
    assert plan.request_kwargs


@pytest.mark.parametrize("mode", ["openai_json_schema", "anthropic_tool", "gemini"])
def test_plan_falls_back_rather_than_raising_on_an_inexpressible_schema(mode: str) -> None:
    """A caller asked for an extraction, not for a particular transport."""

    class Bag(BaseModel):
        payload: dict[str, Any]

    plan = build_schema_plan(Bag, mode)  # type: ignore[arg-type]
    assert plan.native is False
    assert plan.request_kwargs == {}
    assert plan.reason


def test_plan_declines_the_anthropic_tool_path_for_streaming() -> None:
    """A forced tool call streams as tool-call input, so text-delta parsing sees nothing."""
    assert build_schema_plan(Person, "anthropic_tool").native is True
    streaming = build_schema_plan(Person, "anthropic_tool", streaming=True)
    assert streaming.native is False
    assert "tool" in (streaming.reason or "")


def test_plan_keeps_the_other_native_modes_while_streaming() -> None:
    for mode in ("openai_json_schema", "gemini", "ollama"):
        assert build_schema_plan(Person, mode, streaming=True).native is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-provider request bodies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_sends_a_strict_json_schema_response_format() -> None:
    pytest.importorskip("openai")
    from openai import AsyncOpenAI

    from actants.llm.openai_provider import OpenAIProvider

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": PERSON_JSON},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AsyncOpenAI(api_key="k", http_client=http)
        llm = LLM(provider=OpenAIProvider(client=client), model="gpt-4o", tracing=False)
        person = await llm.extract("who?", Person)

    assert person == Person.model_validate_json(PERSON_JSON)
    assert llm.last_schema_plan() is not None and llm.last_schema_plan().native  # type: ignore[union-attr]

    fmt = captured["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["address"]["properties"]["city"]["type"] == "string"


@pytest.mark.asyncio
async def test_gemini_sends_response_schema_inside_generation_config() -> None:
    from actants.llm.gemini_provider import GeminiProvider

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": PERSON_JSON}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = GeminiProvider(api_key="k", client=http)
        llm = LLM(provider=provider, model="gemini-2.5-flash", tracing=False)
        person = await llm.extract("who?", Person)

    assert person.name == "Ada"
    config = captured["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["type"] == "OBJECT"
    assert config["responseSchema"]["properties"]["age"]["type"] == "INTEGER"


@pytest.mark.asyncio
async def test_ollama_sends_the_whole_schema_as_format() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": PERSON_JSON},
                "done_reason": "stop",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = LLM(provider=OllamaProvider(client=http), model="llama3.2", tracing=False)
        person = await llm.extract("who?", Person)

    assert person.address.city == "London"
    fmt = captured["body"]["format"]
    # Not the bare string "json" — a whole schema, with nested models intact.
    assert fmt["type"] == "object"
    assert "Address" in fmt["$defs"]


@pytest.mark.asyncio
async def test_anthropic_forces_a_single_tool_whose_input_schema_is_the_target() -> None:
    pytest.importorskip("anthropic")
    from actants.llm.anthropic_provider import AnthropicProvider

    captured: dict[str, Any] = {}
    arguments = json.loads(PERSON_JSON)

    class FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _anthropic_response(arguments)

    class FakeClient:
        messages = FakeMessages()

    llm = LLM(
        provider=AnthropicProvider(client=FakeClient()),  # type: ignore[arg-type]
        model="claude-opus-5",
        tracing=False,
    )
    person = await llm.extract("who?", Person)

    assert person == Person.model_validate_json(PERSON_JSON)
    assert captured["tool_choice"] == {"type": "tool", "name": ANTHROPIC_EXTRACT_TOOL}
    # What reaches the SDK is Anthropic's own tool shape, built by the provider from the
    # ToolSpec the plan produced — `input_schema`, not `parameters`.
    (tool,) = captured["tools"]
    assert tool["name"] == ANTHROPIC_EXTRACT_TOOL
    schema = tool["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["address"]["properties"]["city"]["type"] == "string"
    assert "$defs" not in schema


def _anthropic_response(arguments: dict[str, Any]) -> Any:
    """The shape AnthropicProvider.complete reads: a tool_use block plus usage."""

    class Block:
        type = "tool_use"
        id = "tu_1"
        name = ANTHROPIC_EXTRACT_TOOL
        input = arguments

    class Usage:
        input_tokens = 1
        output_tokens = 1

    class Response:
        content = [Block()]
        usage = Usage()
        stop_reason = "tool_use"

    return Response()


@pytest.mark.asyncio
async def test_anthropic_extraction_errors_when_the_forced_tool_is_not_called() -> None:
    """tool_choice makes this a provider bug, so it must surface rather than parse text."""
    pytest.importorskip("anthropic")
    from actants.llm.anthropic_provider import AnthropicProvider

    class FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            class Usage:
                input_tokens = 1
                output_tokens = 1

            class Response:
                content: list[Any] = []
                usage = Usage()
                stop_reason = "end_turn"

            return Response()

    class FakeClient:
        messages = FakeMessages()

    llm = LLM(
        provider=AnthropicProvider(client=FakeClient()),  # type: ignore[arg-type]
        model="claude-opus-5",
        tracing=False,
    )
    with pytest.raises(ValueError, match="Failed to extract Person"):
        await llm.extract("who?", Person, max_repairs=0)


# ---------------------------------------------------------------------------
# Provider capability declarations
# ---------------------------------------------------------------------------


def test_built_in_providers_declare_the_right_modes() -> None:
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    from actants.llm.anthropic_provider import AnthropicProvider
    from actants.llm.gemini_provider import GeminiProvider
    from actants.llm.openai_provider import OpenAIProvider

    assert OpenAIProvider.native_schema_mode == "openai_json_schema"
    assert AnthropicProvider.native_schema_mode == "anthropic_tool"
    assert GeminiProvider.native_schema_mode == "gemini"
    assert OllamaProvider.native_schema_mode == "ollama"


def test_openai_compatible_hosts_declare_native_support_individually() -> None:
    """Speaking the OpenAI wire format does not imply implementing json_schema.

    DeepSeek rejects `json_schema` outright; Groq accepts `strict` and honours it only
    on the gpt-oss models, which fails *open* — an unconstrained extraction reporting
    itself as native. Both must therefore use the prompt path.
    """
    pytest.importorskip("openai")
    from actants.llm.groq_provider import GroqProvider
    from actants.llm.mistral_provider import MistralProvider
    from actants.llm.openai_compatible import (
        NO_NATIVE_SCHEMA,
        OPENAI_COMPATIBLE_PROVIDERS,
        CerebrasProvider,
        DeepSeekProvider,
        openai_compatible_provider,
    )

    assert DeepSeekProvider.native_schema_mode == "none"
    assert GroqProvider.native_schema_mode == "none"
    assert MistralProvider.native_schema_mode == "openai_json_schema"
    assert CerebrasProvider.native_schema_mode == "openai_json_schema"

    # Every generated class agrees with the decline table, so adding a row is enough.
    for name, entry in OPENAI_COMPATIBLE_PROVIDERS.items():
        cls = openai_compatible_provider(name, *entry)
        expected = "none" if name in NO_NATIVE_SCHEMA else "openai_json_schema"
        assert cls.native_schema_mode == expected, name


def test_a_custom_provider_gets_the_prompt_path_by_default() -> None:
    """The safe default: a third-party provider must not be assumed to constrain."""

    class Custom(BaseLLMProvider):
        name = "custom"

        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):  # type: ignore[no-untyped-def]
            return CompletionResult(content="{}", model=model, provider=self.name)

        async def health(self) -> bool:
            return True

    assert Custom.native_schema_mode == "none"


# ---------------------------------------------------------------------------
# Fallback behaviour through the client
# ---------------------------------------------------------------------------


class RecordingProvider(BaseLLMProvider):
    """Records the kwargs and messages each call received, and replays scripted text."""

    name = "recording"
    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, responses: list[str], mode: str = "none") -> None:
        self._responses = list(responses)
        self.native_schema_mode = mode  # type: ignore[assignment]
        self.calls: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        self.kwargs.append(dict(kwargs))
        content = self._responses.pop(0)
        calls: list[ToolCall] = []
        if self.native_schema_mode == "anthropic_tool":
            calls = [ToolCall(id="t1", name=ANTHROPIC_EXTRACT_TOOL, arguments=json.loads(content))]
            content = ""
        return CompletionResult(content=content, model=model, provider=self.name, tool_calls=calls)

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        self.kwargs.append(dict(kwargs))
        for chunk in self._responses.pop(0):
            yield TextDelta(text=chunk)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_native_path_omits_the_schema_from_the_prompt() -> None:
    """The decoder is enforcing the schema, so repeating it in the prompt is waste."""
    provider = RecordingProvider([PERSON_JSON], mode="ollama")
    llm = LLM(provider=provider, model="m", tracing=False)
    await llm.extract("who?", Person)

    assert not [m for m in provider.calls[0] if m.role == "system"]
    assert "format" in provider.kwargs[0]


@pytest.mark.asyncio
async def test_fallback_path_puts_the_schema_in_the_prompt_and_sends_no_kwargs() -> None:
    provider = RecordingProvider([PERSON_JSON], mode="none")
    llm = LLM(provider=provider, model="m", tracing=False)
    person = await llm.extract("who?", Person)

    assert person.name == "Ada"
    (system,) = [m for m in provider.calls[0] if m.role == "system"]
    assert "JSON Schema" in system.content
    assert provider.kwargs[0] == {}


@pytest.mark.asyncio
async def test_a_schema_the_provider_cannot_express_falls_back_on_a_native_provider() -> None:
    """The headline fallback: native provider, inexpressible schema, prompt path taken."""

    class Bag(BaseModel):
        payload: dict[str, Any]

    provider = RecordingProvider(['{"payload": {"a": 1}}'], mode="openai_json_schema")
    llm = LLM(provider=provider, model="m", tracing=False)
    bag = await llm.extract("give me a bag", Bag)

    assert bag.payload == {"a": 1}
    plan = llm.last_schema_plan()
    assert plan is not None and plan.native is False
    assert plan.mode == "openai_json_schema"  # the capability, not what ran
    assert "additionalProperties" in (plan.reason or "")
    assert "response_format" not in provider.kwargs[0]
    assert [m for m in provider.calls[0] if m.role == "system"]


@pytest.mark.asyncio
async def test_extract_preserves_a_caller_system_prompt_on_both_paths() -> None:
    for mode, expect_guide in (("ollama", False), ("none", True)):
        provider = RecordingProvider([PERSON_JSON], mode=mode)
        llm = LLM(provider=provider, model="m", tracing=False)
        await llm.extract("who?", Person, system="Be terse.")
        (system,) = [m for m in provider.calls[0] if m.role == "system"]
        assert system.content.startswith("Be terse.")
        assert ("JSON Schema" in system.content) is expect_guide


@pytest.mark.asyncio
async def test_last_schema_plan_is_none_before_any_extraction() -> None:
    llm = LLM(provider=RecordingProvider([]), model="m", tracing=False)
    assert llm.last_schema_plan() is None


@pytest.mark.asyncio
async def test_native_extraction_needs_no_repair_even_with_repairs_disabled() -> None:
    """max_repairs keeps its documented meaning; the native path just never uses it."""
    provider = RecordingProvider([PERSON_JSON], mode="ollama")
    llm = LLM(provider=provider, model="m", tracing=False)
    person = await llm.extract("who?", Person, max_repairs=0)
    assert person.name == "Ada"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_strict_mode_null_for_a_defaulted_field_validates_in_one_call() -> None:
    """Regression: the strict rewrite produced output pydantic then rejected.

    ``priority: int = 3`` is widened to ``["integer", "null"]`` and marked required, so a
    provider with real constrained decoding may answer ``null``. That used to fail
    validation, burn a repair, and raise — despite the response obeying the schema
    actants itself sent. Asserted on the validated result and the call count, which is
    what "a schema-valid response cannot fail to parse" actually promises.
    """

    class Task(BaseModel):
        name: str
        priority: int = 3
        tags: list[str] = Field(default_factory=list)

    provider = RecordingProvider(
        ['{"name": "ship it", "priority": null, "tags": null}'],
        mode="openai_json_schema",
    )
    llm = LLM(provider=provider, model="m", tracing=False)
    task = await llm.extract("make a task", Task, max_repairs=0)

    assert task == Task(name="ship it", priority=3, tags=[])
    assert len(provider.calls) == 1, "a schema-valid response must not need a repair"


@pytest.mark.asyncio
async def test_a_genuinely_optional_field_keeps_its_null_on_the_strict_path() -> None:
    """The other half: ``X | None`` still means None, not "fall back to the default"."""
    provider = RecordingProvider(
        [
            '{"name": "Ada", "age": 36, "address": {"street": "1 Main", "city": "London"}, '
            '"nickname": null}'
        ],
        mode="openai_json_schema",
    )
    llm = LLM(provider=provider, model="m", tracing=False)
    person = await llm.extract("who?", Person, max_repairs=0)
    assert person.nickname is None


@pytest.mark.asyncio
async def test_repair_still_runs_on_the_prompt_path() -> None:
    provider = RecordingProvider(["not json", PERSON_JSON], mode="none")
    llm = LLM(provider=provider, model="m", tracing=False)
    person = await llm.extract("who?", Person, max_repairs=1)
    assert person.name == "Ada"
    assert len(provider.calls) == 2
    assert any("did not parse" in m.content for m in provider.calls[1])


# ---------------------------------------------------------------------------
# extract_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_stream_uses_a_text_native_mode_and_still_yields_partials() -> None:
    chunks = ['{"name": "Ada", "age": 36, "address"', ': {"street": "1 Main", "city": "London"}}']
    provider = RecordingProvider([chunks], mode="ollama")  # type: ignore[list-item]
    llm = LLM(provider=provider, model="m", tracing=False)

    seen = [p async for p in llm.extract_stream("who?", Person)]
    plan = llm.last_schema_plan()
    assert plan is not None and plan.native is True
    assert "format" in provider.kwargs[0]
    assert seen[-1] == Person.model_validate_json(PERSON_JSON)


@pytest.mark.asyncio
async def test_extract_stream_declines_the_anthropic_tool_path() -> None:
    """Its JSON is tool-call input, so a text stream would see nothing to parse."""
    chunks = [PERSON_JSON]
    provider = RecordingProvider([chunks], mode="anthropic_tool")  # type: ignore[list-item]
    llm = LLM(provider=provider, model="m", tracing=False)

    seen = [p async for p in llm.extract_stream("who?", Person)]
    plan = llm.last_schema_plan()
    assert plan is not None and plan.native is False
    assert provider.kwargs[0] == {}
    assert [m for m in provider.calls[0] if m.role == "system"]
    assert seen[-1].name == "Ada"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_new_symbols_are_exported_from_the_top_level() -> None:
    import actants

    for name in (
        "SchemaPlan",
        "NativeSchemaMode",
        "NATIVE_SCHEMA_MODES",
        "UnsupportedSchemaError",
        "build_schema_plan",
        "to_strict_schema",
        "to_gemini_schema",
    ):
        assert name in actants.__all__, name
        assert getattr(actants, name) is not None


def test_unsupported_schema_error_is_in_the_actants_hierarchy() -> None:
    from actants import ActantsError

    assert issubclass(UnsupportedSchemaError, ActantsError)
    assert issubclass(UnsupportedSchemaError, ValueError)
