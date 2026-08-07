"""Normalization of provider-specific stop reasons into one portable vocabulary.

Every provider spells "the model stopped because it ran out of tokens" differently:
OpenAI says ``"length"``, Anthropic says ``"max_tokens"``, Gemini says ``"MAX_TOKENS"``,
Ollama says ``"length"``. A caller holding a provider-agnostic
:class:`~actants.llm.base.CompletionResult` cannot branch on that without writing the
union of every provider's vocabulary — which is exactly the coupling the result type
exists to remove.

:data:`FinishReason` is therefore a closed :class:`~typing.Literal` of six canonical
values, and :func:`normalize_finish_reason` maps each provider's string onto it. The
provider's original string is never discarded: it is preserved verbatim on
:attr:`~actants.llm.base.CompletionResult.raw_finish_reason`, so anything that needs the
exact wire value — logging, a provider-specific workaround, telemetry — still has it.

**Unknown values never raise.** Providers add stop reasons on their own schedule; Gemini
alone has grown ``MALFORMED_FUNCTION_CALL``, ``BLOCKLIST``, ``SPII``, ``IMAGE_SAFETY``,
and more since launch. A closed mapping that raised on an unrecognized string would turn
"the provider shipped a new enum member" into a crash in the middle of a user's
completion. Anything unrecognized maps to ``"unknown"`` and keeps its raw string, so the
information is still there and the call still succeeds.
"""

from __future__ import annotations

from typing import Literal, get_args

#: The canonical, provider-independent reasons a completion stopped.
#:
#: * ``"stop"`` — the model finished normally, including hitting a caller-supplied stop
#:   sequence. This is the success case.
#: * ``"length"`` — generation was cut off by a token limit. The output is truncated.
#: * ``"tool_calls"`` — the model stopped in order to call tools. Dispatch them and
#:   continue the conversation.
#: * ``"content_filter"`` — the provider's safety or policy layer blocked the output.
#:   The content is absent or partial, and retrying the same prompt will usually fail
#:   the same way.
#: * ``"error"`` — the provider reported a generation-side failure, such as a tool call
#:   it could not render as valid JSON.
#: * ``"unknown"`` — no reason was reported, or the provider sent a value this version of
#:   actants does not recognize. Check
#:   :attr:`~actants.llm.base.CompletionResult.raw_finish_reason` for what it actually
#:   said.
FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
    "unknown",
]

#: Runtime mirror of :data:`FinishReason`, for validation and for tests that assert the
#: mapping tables only ever produce a canonical value.
FINISH_REASONS: tuple[FinishReason, ...] = get_args(FinishReason)

#: The value used when a provider reports nothing, or reports something unrecognized.
UNKNOWN_FINISH_REASON: FinishReason = "unknown"


#: OpenAI's ``choice.finish_reason``, from
#: ``openai.types.chat.chat_completion.Choice``. ``function_call`` is the deprecated
#: predecessor of ``tool_calls`` and means the same thing to a caller.
#:
#: This table also serves every OpenAI-compatible provider — Groq, Mistral, xAI,
#: DeepSeek, Together, Fireworks, OpenRouter, Cerebras, Perplexity — because they return
#: OpenAI's response shape verbatim, which is the entire premise of
#: :mod:`actants.llm.openai_compatible`.
_OPENAI: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}

#: Anthropic's ``stop_reason``, from ``anthropic.types.stop_reason.StopReason``.
#:
#: ``pause_turn`` marks a long-running turn the caller is expected to continue by sending
#: the response back — the model has not failed and has not finished, so it maps to
#: ``"stop"`` rather than ``"error"``: the turn ended cleanly and control is with the
#: caller. ``refusal`` is the model declining on policy grounds, which is what
#: ``content_filter`` means everywhere else.
_ANTHROPIC: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

#: Gemini's ``candidates[].finishReason``. Google adds members to this enum regularly, so
#: this table is a best effort over the documented set and anything new lands on
#: ``"unknown"`` with its raw value intact rather than crashing the call.
#:
#: The safety family (``SAFETY``, ``RECITATION``, ``BLOCKLIST``, ``PROHIBITED_CONTENT``,
#: ``SPII``, ``IMAGE_SAFETY``, ``MODEL_ARMOR``, ...) all mean "policy blocked this", which
#: is ``content_filter``. ``MALFORMED_FUNCTION_CALL`` and its siblings are the model
#: failing to produce a usable tool call, which is ``error``.
_GEMINI: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    # Safety / policy blocks.
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "IMAGE_RECITATION": "content_filter",
    "MODEL_ARMOR": "content_filter",
    # Generation-side failures.
    "MALFORMED_FUNCTION_CALL": "error",
    "UNEXPECTED_TOOL_CALL": "error",
    "TOO_MANY_TOOL_CALLS": "error",
    # Explicitly-unspecified and catch-all members. Mapping these to ``"unknown"`` is
    # the same answer the fallback would give, but stating it here documents that they
    # were considered rather than missed.
    "FINISH_REASON_UNSPECIFIED": "unknown",
    "OTHER": "unknown",
    "LANGUAGE": "unknown",
    "NO_IMAGE": "unknown",
    "IMAGE_OTHER": "unknown",
}

#: Ollama's ``done_reason``. Verified against a live Ollama (0.x, ``/api/chat``): a normal
#: completion and a stop-sequence hit both report ``"stop"``, and ``num_predict`` running
#: out reports ``"length"`` — the OpenAI spelling, not Anthropic's.
_OLLAMA: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "load": "unknown",
    "unload": "unknown",
}

#: provider name -> that provider's mapping table.
#:
#: Providers absent from this table fall back to the OpenAI vocabulary, which is correct
#: for every OpenAI-compatible provider and is also the most likely shape for a
#: third-party provider written against :class:`~actants.llm.base.BaseLLMProvider`.
_TABLES: dict[str, dict[str, FinishReason]] = {
    "openai": _OPENAI,
    "anthropic": _ANTHROPIC,
    "gemini": _GEMINI,
    "ollama": _OLLAMA,
}


def normalize_finish_reason(provider: str, raw: str | None) -> FinishReason:
    """Map ``raw`` from ``provider`` onto a canonical :data:`FinishReason`.

    ``None`` and the empty string mean the provider reported nothing, which is
    ``"unknown"``. An unrecognized value is also ``"unknown"`` — never an exception —
    because providers extend these enums without warning and a completion that already
    succeeded must not be turned into a crash by a string actants has not seen before.

    Lookup is case-insensitive on a second pass, so a provider that switches between
    ``"STOP"`` and ``"stop"`` maps identically either way.

    Example::

        >>> normalize_finish_reason("anthropic", "end_turn")
        'stop'
        >>> normalize_finish_reason("gemini", "MAX_TOKENS")
        'length'
        >>> normalize_finish_reason("openai", "some_new_reason_2027")
        'unknown'
    """
    if not raw:
        return UNKNOWN_FINISH_REASON
    table = _TABLES.get(provider.lower(), _OPENAI)
    found = table.get(raw)
    if found is not None:
        return found
    # Second pass, case-insensitively: Gemini keys are upper-case and OpenAI's are lower,
    # and a compatible provider occasionally returns the other casing.
    lowered = raw.lower()
    for key, value in table.items():
        if key.lower() == lowered:
            return value
    return UNKNOWN_FINISH_REASON


__all__ = [
    "FINISH_REASONS",
    "UNKNOWN_FINISH_REASON",
    "FinishReason",
    "normalize_finish_reason",
]
