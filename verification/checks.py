"""The five checks each provider is put through.

Every check returns a :class:`CheckResult` rather than raising, so one provider failing
a check never stops the rest of the run — the whole value of the harness is the full
matrix, including the failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from actants.cost.pricing import estimate_cost_or_none, is_priced
from actants.llm.base import FinishDelta, TextDelta, ToolSpec, UsageDelta
from actants.llm.client import LLM
from actants.tools.registry import ToolRegistry

#: Every prompt here is deliberately tiny and asks for a short answer. The harness
#: verifies wire formats, and a long generation costs more without exercising anything
#: the first twenty tokens did not.
MAX_TOKENS = 64


#: Substrings identifying an error that is about the *account*, not the integration —
#: an exhausted balance, a revoked key, a model the org cannot reach. Reporting these as
#: FAIL would claim actants is broken against a provider it was never allowed to call,
#: which is exactly the false confidence this harness exists to remove.
_ACCOUNT_ERROR_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing",
    "no credits remaining",
    "exceeded your current quota",
    "invalid_api_key",
    "authentication",
    "model_not_found",
    "does not have access to model",
)


def classify_error(exc: Exception) -> tuple[str, str]:
    """Return ``(status, detail)`` for an exception raised by a live call."""
    detail = f"{type(exc).__name__}: {exc}"
    lowered = detail.lower()
    if any(marker in lowered for marker in _ACCOUNT_ERROR_MARKERS):
        return "blocked", detail
    return "fail", detail


@dataclass
class CheckResult:
    name: str
    #: "pass" | "fail" | "skip" | "blocked". ``blocked`` is an account-level refusal
    #: (no credits, revoked key) — the integration was never exercised, so it is neither
    #: a pass nor evidence of a defect.
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passed(cls, name: str, detail: str = "", **data: Any) -> CheckResult:
        return cls(name, "pass", detail, data)

    @classmethod
    def failed(cls, name: str, detail: str, **data: Any) -> CheckResult:
        return cls(name, "fail", detail, data)

    @classmethod
    def skipped(cls, name: str, detail: str) -> CheckResult:
        return cls(name, "skip", detail)

    @classmethod
    def from_exception(cls, name: str, exc: Exception, **data: Any) -> CheckResult:
        status, detail = classify_error(exc)
        return cls(name, status, detail, data)


class Address(BaseModel):
    city: str
    country: str


class Person(BaseModel):
    """Nested on purpose: a flat model would not exercise $defs/$ref rewriting, which is
    where every native-schema dialect differs most."""

    name: str
    age: int
    address: Address
    nicknames: list[str] = Field(default_factory=list)


async def check_complete(llm: LLM, model: str) -> CheckResult:
    try:
        r = await llm.complete(
            "Reply with exactly the word: pong",
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:
        return CheckResult.from_exception("complete", exc)

    problems = []
    if not r.content.strip():
        problems.append("empty content")
    if r.provider != llm.provider.name:
        problems.append(f"provider={r.provider!r} != {llm.provider.name!r}")
    if r.finish_reason == "unknown":
        problems.append(f"finish_reason unknown (raw={r.raw_finish_reason!r})")
    if r.usage.prompt_tokens <= 0:
        problems.append("no prompt tokens reported")
    if r.latency_ms <= 0:
        problems.append("no latency recorded")
    if problems:
        return CheckResult.failed("complete", "; ".join(problems))
    return CheckResult.passed(
        "complete",
        f"finish_reason={r.finish_reason} (raw={r.raw_finish_reason!r}) "
        f"tokens={r.usage.prompt_tokens}+{r.usage.completion_tokens}",
        finish_reason=r.finish_reason,
        raw_finish_reason=r.raw_finish_reason,
        prompt_tokens=r.usage.prompt_tokens,
        completion_tokens=r.usage.completion_tokens,
    )


async def check_stream(llm: LLM, model: str) -> CheckResult:
    """Text deltas, then usage at stream end, then finish.

    The usage-at-end assertion is the one most likely to be wrong per provider: it
    requires the provider to opt into usage reporting on a stream (OpenAI needs
    ``stream_options``), and a provider that does not is silently uncosted.
    """
    deltas: list[str] = []
    usage_event: UsageDelta | None = None
    finish_event: FinishDelta | None = None
    order: list[str] = []
    try:
        async for event in llm.stream_events(
            "Count: one two three", model=model, max_tokens=MAX_TOKENS, temperature=0.0
        ):
            order.append(event.type)
            if isinstance(event, TextDelta):
                deltas.append(event.text)
            elif isinstance(event, UsageDelta):
                usage_event = event
            elif isinstance(event, FinishDelta):
                finish_event = event
    except Exception as exc:
        return CheckResult.from_exception("stream", exc)

    problems = []
    if not deltas:
        problems.append("no text deltas")
    if usage_event is None:
        problems.append("no UsageDelta at stream end")
    elif usage_event.usage.total_tokens <= 0:
        problems.append("UsageDelta reported zero tokens")
    if finish_event is None:
        problems.append("no FinishDelta")
    elif finish_event.reason == "unknown":
        problems.append(f"finish reason unknown (raw={finish_event.raw_reason!r})")
    if order and order[-1] != "finish":
        problems.append(f"terminal event was {order[-1]!r}, not 'finish'")
    if problems:
        return CheckResult.failed("stream", "; ".join(problems))

    assert usage_event is not None and finish_event is not None
    return CheckResult.passed(
        "stream",
        f"{len(deltas)} deltas, usage at end "
        f"({usage_event.usage.prompt_tokens}+{usage_event.usage.completion_tokens}), "
        f"finish={finish_event.reason}",
        text_deltas=len(deltas),
        usage_at_end=True,
        prompt_tokens=usage_event.usage.prompt_tokens,
        completion_tokens=usage_event.usage.completion_tokens,
        finish_reason=finish_event.reason,
    )


async def check_stream_matches_complete(llm: LLM, model: str) -> CheckResult:
    """Concatenated deltas must equal the content a non-streamed call reports.

    Run at temperature 0 against a prompt with one sane answer, so a mismatch means the
    stream assembler dropped or duplicated a chunk rather than the model being creative.
    """
    prompt = "Repeat exactly, with no other words: alpha bravo charlie"
    try:
        streamed = "".join(
            [
                chunk
                async for chunk in llm.stream(
                    prompt, model=model, max_tokens=MAX_TOKENS, temperature=0.0
                )
            ]
        )
        completed = await llm.complete(prompt, model=model, max_tokens=MAX_TOKENS, temperature=0.0)
    except Exception as exc:
        return CheckResult.from_exception("stream_matches_complete", exc)

    if not streamed.strip():
        return CheckResult.failed("stream_matches_complete", "stream produced no text")
    if streamed.strip() != completed.content.strip():
        return CheckResult.failed(
            "stream_matches_complete",
            f"streamed {streamed.strip()[:80]!r} != completed {completed.content.strip()[:80]!r}",
            streamed=streamed.strip()[:200],
            completed=completed.content.strip()[:200],
        )
    return CheckResult.passed("stream_matches_complete", f"both produced {streamed.strip()[:40]!r}")


async def check_tool_call(llm: LLM, model: str) -> CheckResult:
    if not llm.provider.supports_tool_calls:
        return CheckResult.skipped("tool_call", "provider declares supports_tool_calls=False")

    calls: list[str] = []

    async def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append(f"add({a},{b})")
        return a + b

    registry = ToolRegistry()
    registry.register_function("add", "Add two integers together", add)

    try:
        result = await llm.run_agent(
            "What is 21 plus 21? Use the add tool, then state the number.",
            registry,
            model=model,
            temperature=0.0,
            max_steps=4,
        )
    except Exception as exc:
        return CheckResult.from_exception("tool_call", exc)

    if not calls:
        return CheckResult.failed(
            "tool_call", f"model never invoked the tool; answered {result.content[:80]!r}"
        )
    if "42" not in result.content:
        return CheckResult.failed(
            "tool_call",
            f"tool ran ({calls}) but final answer lacks 42: {result.content[:80]!r}",
            tool_invocations=calls,
        )
    return CheckResult.passed("tool_call", f"round-tripped {calls}", tool_invocations=calls)


async def check_streaming_tool_call(llm: LLM, model: str) -> CheckResult:
    if not llm.provider.supports_streaming_tools:
        return CheckResult.skipped(
            "streaming_tool_call", "provider declares supports_streaming_tools=False"
        )
    spec = ToolSpec(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    seen = []
    try:
        async for event in llm.stream_events(
            "What is the weather in Paris? Use the get_weather tool.",
            model=model,
            tools=[spec],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        ):
            if event.type == "tool_call":
                seen.append(event.tool_call)
    except Exception as exc:
        return CheckResult.from_exception("streaming_tool_call", exc)

    if not seen:
        return CheckResult.failed(
            "streaming_tool_call", "no ToolCallDelta emitted despite supports_streaming_tools=True"
        )
    call = seen[0]
    if call.name != "get_weather":
        return CheckResult.failed("streaming_tool_call", f"unexpected tool {call.name!r}")
    if not call.arguments.get("city"):
        return CheckResult.failed(
            "streaming_tool_call", f"tool call arrived with no city argument: {call.arguments!r}"
        )
    return CheckResult.passed(
        "streaming_tool_call",
        f"{call.name}({call.arguments})",
        tool_name=call.name,
        arguments=call.arguments,
    )


async def check_structured_output(llm: LLM, model: str, expected_mode: str) -> CheckResult:
    """The highest-value check: does the native-schema wire format actually work?

    Asserts three things — that the extraction parses, that the plan taken matches what
    the provider declares, and that a *native* plan really did go native. A provider that
    declares a native mode but falls back to the prompt path is reported as a failure,
    because the guarantee the feature sells is that invalid output is impossible.
    """
    try:
        person = await llm.extract(
            "Ada Lovelace is 36 and lives in London, United Kingdom. "
            "Friends call her Ada and Countess.",
            Person,
            model=model,
            temperature=0.0,
        )
    except Exception as exc:
        plan = llm.last_schema_plan()
        return CheckResult.from_exception(
            "structured_output",
            exc,
            planned_native=plan.native if plan else None,
            planned_mode=plan.mode if plan else None,
            plan_reason=plan.reason if plan else None,
        )

    plan = llm.last_schema_plan()
    if plan is None:
        return CheckResult.failed("structured_output", "extract() recorded no schema plan")

    problems = []
    if plan.mode != expected_mode:
        problems.append(f"plan mode {plan.mode!r} != declared {expected_mode!r}")
    if expected_mode != "none" and not plan.native:
        problems.append(f"declared native mode {expected_mode!r} but fell back: {plan.reason}")
    if not person.name or person.address.city.lower() != "london":
        problems.append(f"implausible extraction: {person.model_dump()}")

    detail = (
        f"path={'native' if plan.native else 'prompt'} mode={plan.mode} -> {person.model_dump()}"
    )
    if problems:
        return CheckResult.failed(
            "structured_output",
            "; ".join(problems) + f" | {detail}",
            native=plan.native,
            mode=plan.mode,
            reason=plan.reason,
        )
    return CheckResult.passed(
        "structured_output",
        detail,
        native=plan.native,
        mode=plan.mode,
        extracted=person.model_dump(),
    )


async def check_cost_attribution(llm: LLM, model: str) -> CheckResult:
    """Recompute cost from the provider's own reported usage and the published price.

    This is the check that catches a provider whose ``cost_usd`` was computed against a
    different model string than the one billed, or that never got wired to
    ``estimate_cost`` at all.
    """
    provider = llm.provider.name
    try:
        r = await llm.complete("Say: ok", model=model, max_tokens=MAX_TOKENS, temperature=0.0)
    except Exception as exc:
        return CheckResult.from_exception("cost_attribution", exc)

    priced = is_priced(provider, model)
    expected = estimate_cost_or_none(
        provider, model, r.usage.prompt_tokens, r.usage.completion_tokens
    )

    if not priced:
        if r.cost_usd != 0.0:
            return CheckResult.failed(
                "cost_attribution",
                f"model is unpriced but cost_usd={r.cost_usd} (should floor to 0.0)",
            )
        return CheckResult.passed(
            "cost_attribution",
            f"unpriced model, cost_usd floored to 0.0 "
            f"({r.usage.prompt_tokens}+{r.usage.completion_tokens} tokens uncosted)",
            priced=False,
            cost_usd=0.0,
        )

    assert expected is not None
    # Exact within float noise: both sides run the same arithmetic on the same integers,
    # so any real discrepancy is a wiring bug, not a rounding one.
    if abs(r.cost_usd - expected) > 1e-12:
        return CheckResult.failed(
            "cost_attribution",
            f"cost_usd={r.cost_usd!r} but usage {r.usage.prompt_tokens}+"
            f"{r.usage.completion_tokens} at published price = {expected!r}",
            priced=True,
            reported=r.cost_usd,
            recomputed=expected,
        )
    return CheckResult.passed(
        "cost_attribution",
        f"cost_usd={r.cost_usd:.8f} matches {r.usage.prompt_tokens}+"
        f"{r.usage.completion_tokens} tokens at published price",
        priced=True,
        cost_usd=r.cost_usd,
    )


async def timed(coro: Any) -> tuple[CheckResult, float]:
    start = time.perf_counter()
    result = await coro
    return result, (time.perf_counter() - start) * 1000
