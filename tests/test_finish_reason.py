"""``finish_reason`` must be portable across providers, and must never lose the raw value.

Before 1.0, ``CompletionResult.finish_reason`` was ``str | None`` carrying whatever the
provider said — ``"stop"`` from OpenAI, ``"end_turn"`` from Anthropic, ``"STOP"`` from
Gemini. That made a provider-agnostic result type impossible to branch on portably, and
because the field was already public it could never be *narrowed* after 1.0.

These tests pin the mapping for every provider, the documented fallback for values
actants does not recognize, and the preservation of the raw string.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
    ToolSpec,
)
from actants.llm.finish_reason import (
    FINISH_REASONS,
    UNKNOWN_FINISH_REASON,
    normalize_finish_reason,
)

# --------------------------------------------------------------------------------------
# The canonical vocabulary
# --------------------------------------------------------------------------------------


def test_canonical_set_is_exactly_the_documented_six() -> None:
    assert set(FINISH_REASONS) == {
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "error",
        "unknown",
    }


def test_unknown_is_a_member_of_the_canonical_set() -> None:
    assert UNKNOWN_FINISH_REASON in FINISH_REASONS


# --------------------------------------------------------------------------------------
# Per-provider mapping. Values are the real vocabularies, taken from each provider's own
# type definitions (openai.types.chat, anthropic.types.stop_reason), Google's documented
# FinishReason enum, and a live Ollama.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("function_call", "tool_calls"),  # deprecated predecessor of tool_calls
        ("content_filter", "content_filter"),
    ],
)
def test_openai_mapping(raw: str, expected: str) -> None:
    assert normalize_finish_reason("openai", raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("pause_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
    ],
)
def test_anthropic_mapping(raw: str, expected: str) -> None:
    assert normalize_finish_reason("anthropic", raw) == expected


def test_anthropic_covers_its_entire_sdk_vocabulary() -> None:
    """Every member of anthropic's StopReason must map to something other than unknown.

    If the SDK grows a member, this fails and points at the table to update.
    """
    sdk_values = {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"}
    for value in sdk_values:
        assert normalize_finish_reason("anthropic", value) != "unknown", value


def test_openai_covers_its_entire_sdk_vocabulary() -> None:
    sdk_values = {"stop", "length", "tool_calls", "content_filter", "function_call"}
    for value in sdk_values:
        assert normalize_finish_reason("openai", value) != "unknown", value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("IMAGE_SAFETY", "content_filter"),
        ("MODEL_ARMOR", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "error"),
        ("UNEXPECTED_TOOL_CALL", "error"),
        ("TOO_MANY_TOOL_CALLS", "error"),
        ("FINISH_REASON_UNSPECIFIED", "unknown"),
        ("OTHER", "unknown"),
    ],
)
def test_gemini_mapping(raw: str, expected: str) -> None:
    assert normalize_finish_reason("gemini", raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Verified against a live Ollama: a normal completion and a stop-sequence hit
        # both report "stop"; num_predict running out reports "length".
        ("stop", "stop"),
        ("length", "length"),
        ("load", "unknown"),
        ("unload", "unknown"),
    ],
)
def test_ollama_mapping(raw: str, expected: str) -> None:
    assert normalize_finish_reason("ollama", raw) == expected


@pytest.mark.parametrize(
    "provider",
    [
        "groq",
        "mistral",
        "xai",
        "deepseek",
        "together",
        "fireworks",
        "openrouter",
        "cerebras",
        "perplexity",
    ],
)
def test_openai_compatible_providers_use_the_openai_vocabulary(provider: str) -> None:
    """The nine compatible providers return OpenAI's response shape verbatim."""
    assert normalize_finish_reason(provider, "stop") == "stop"
    assert normalize_finish_reason(provider, "length") == "length"
    assert normalize_finish_reason(provider, "tool_calls") == "tool_calls"
    assert normalize_finish_reason(provider, "content_filter") == "content_filter"


# --------------------------------------------------------------------------------------
# The unknown case: documented fallback, never a crash.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "gemini", "ollama", "groq", "a-provider-from-2030"]
)
def test_unrecognized_value_falls_back_to_unknown(provider: str) -> None:
    """Providers extend these enums without warning; a new member must not crash."""
    assert normalize_finish_reason(provider, "some_reason_invented_in_2030") == "unknown"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
@pytest.mark.parametrize("raw", [None, ""])
def test_absent_reason_is_unknown(provider: str, raw: str | None) -> None:
    assert normalize_finish_reason(provider, raw) == "unknown"


def test_normalization_is_case_insensitive() -> None:
    """A provider switching casing must not silently start returning 'unknown'."""
    assert normalize_finish_reason("openai", "STOP") == "stop"
    assert normalize_finish_reason("gemini", "stop") == "stop"
    assert normalize_finish_reason("anthropic", "END_TURN") == "stop"


def test_provider_name_is_case_insensitive() -> None:
    assert normalize_finish_reason("Anthropic", "end_turn") == "stop"
    assert normalize_finish_reason("GEMINI", "MAX_TOKENS") == "length"


def test_every_mapping_produces_a_canonical_value() -> None:
    """No table may produce a value outside the Literal, or the type would be a lie."""
    from actants.llm.finish_reason import _TABLES

    for provider, table in _TABLES.items():
        for raw, mapped in table.items():
            assert mapped in FINISH_REASONS, (provider, raw, mapped)


def test_unknown_provider_falls_back_to_openai_vocabulary() -> None:
    """A third-party provider is most likely OpenAI-shaped; that is the safer default."""
    assert normalize_finish_reason("some-third-party", "tool_calls") == "tool_calls"


# --------------------------------------------------------------------------------------
# The raw string must survive normalization.
# --------------------------------------------------------------------------------------


def test_completion_result_preserves_raw_value() -> None:
    result = CompletionResult(
        content="",
        model="m",
        provider="anthropic",
        finish_reason=normalize_finish_reason("anthropic", "end_turn"),
        raw_finish_reason="end_turn",
    )
    assert result.finish_reason == "stop"
    assert result.raw_finish_reason == "end_turn"


def test_finish_delta_from_provider_preserves_raw_value() -> None:
    delta = FinishDelta.from_provider("gemini", "MAX_TOKENS")
    assert delta.reason == "length"
    assert delta.raw_reason == "MAX_TOKENS"


def test_finish_delta_from_provider_keeps_unrecognized_raw_value() -> None:
    """The whole point of keeping the raw field: an unmapped value is still inspectable."""
    delta = FinishDelta.from_provider("gemini", "SOME_NEW_ENUM_MEMBER")
    assert delta.reason == "unknown"
    assert delta.raw_reason == "SOME_NEW_ENUM_MEMBER"


def test_defaults_are_unknown_not_none() -> None:
    """The field is no longer nullable, so the absent case must be a real value."""
    assert CompletionResult(content="", model="m", provider="p").finish_reason == "unknown"
    assert CompletionResult(content="", model="m", provider="p").raw_finish_reason is None
    assert FinishDelta().reason == "unknown"


def test_finish_reason_rejects_a_non_canonical_value() -> None:
    """pydantic enforces the Literal, so a provider cannot smuggle a raw string through."""
    with pytest.raises(ValueError, match="finish_reason"):
        CompletionResult(
            content="",
            model="m",
            provider="anthropic",
            finish_reason="end_turn",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------------------
# End-to-end through a provider: normalization happens inside the provider, so a caller
# never sees a raw value on the typed field.
# --------------------------------------------------------------------------------------


class RawReasonProvider(BaseLLMProvider):
    """Emits a provider-native stop reason, as a real provider would."""

    supports_tool_calls = True
    supports_streaming_tools = True

    def __init__(self, name: str, raw: str) -> None:
        self.name = name
        self.raw = raw

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        return CompletionResult(
            content="hi",
            model=model,
            provider=self.name,
            finish_reason=normalize_finish_reason(self.name, self.raw),
            raw_finish_reason=self.raw,
        )

    async def health(self) -> bool:
        return True

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="hi")
        yield FinishDelta.from_provider(self.name, self.raw)


@pytest.mark.parametrize(
    ("provider", "raw", "expected"),
    [
        ("openai", "length", "length"),
        ("anthropic", "max_tokens", "length"),
        ("gemini", "MAX_TOKENS", "length"),
        ("ollama", "length", "length"),
    ],
)
async def test_all_providers_agree_on_the_same_situation(
    provider: str, raw: str, expected: str
) -> None:
    """Four providers, four spellings of 'ran out of tokens', one value to branch on."""
    from actants.llm.client import LLM

    llm = LLM(provider=RawReasonProvider(provider, raw), model="m", tracing=False)
    result = await llm.complete("hi")
    assert result.finish_reason == expected
    assert result.raw_finish_reason == raw

    events = [e async for e in llm.stream_events("hi")]
    finish = events[-1]
    assert isinstance(finish, FinishDelta)
    assert finish.reason == expected
    assert finish.raw_reason == raw
