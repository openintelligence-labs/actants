"""The evaluation harness: scorers, trajectory assertions, and run-to-run deltas.

The contract these pin down:

1. every scorer type does what it says (exact match, contains, predicate, tool-called)
2. a tool-trajectory assertion catches a *wrong argument*, not just a missing call
3. cost and latency deltas between two runs are reported, using the real cost machinery
4. a report is readable for a human and machine-readable for CI
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from actants.agents.agent import Agent
from actants.errors import ActantsError, ScorerError
from actants.graph.state import END
from actants.graph.state_graph import StateGraph
from actants.llm.base import CompletionResult, TokenUsage, ToolCall
from actants.llm.client import LLM
from actants.testing import (
    Contains,
    EvalCase,
    EvalReport,
    EvalSuite,
    ExactMatch,
    FakeLLMProvider,
    Predicate,
    RunOutcome,
    Score,
    ToolCalled,
    ToolsCalledInOrder,
    fake_completion,
    fake_tool_call_completion,
)
from actants.tools.registry import ToolRegistry


def _outcome(output: str = "", **kw: Any) -> RunOutcome:
    return RunOutcome(output=output, **kw)


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=arguments)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def refund(amount: int) -> str:
        return f"refunded {amount}"

    async def lookup(order: str) -> str:
        return f"order {order}"

    registry.register_function("refund", "Refund an order", refund)
    registry.register_function("lookup", "Look an order up", lookup)
    return registry


def _agent(*responses: CompletionResult, tools: ToolRegistry | None = None) -> Agent:
    return Agent(
        llm=LLM(provider=FakeLLMProvider(list(responses)), model="fake", tracing=False),
        tools=tools,
    )


# ---------------------------------------------------------------------------
# 1. Every scorer type
# ---------------------------------------------------------------------------


def test_exact_match_passes_on_an_equal_answer() -> None:
    score = ExactMatch("391").score(_outcome("391"))
    assert score.passed and score.value == 1.0


def test_exact_match_ignores_surrounding_whitespace_by_default() -> None:
    """A model answering "391\\n" has not got the answer wrong."""
    assert ExactMatch("391").score(_outcome("  391\n")).passed


def test_exact_match_can_be_made_strict_about_whitespace() -> None:
    assert not ExactMatch("391", strip=False).score(_outcome(" 391")).passed


def test_exact_match_fails_and_says_what_it_wanted() -> None:
    score = ExactMatch("391").score(_outcome("392"))
    assert not score.passed
    assert "'391'" in score.detail and "'392'" in score.detail


def test_exact_match_can_ignore_case() -> None:
    assert ExactMatch("Berlin", case_sensitive=False).score(_outcome("berlin")).passed
    assert not ExactMatch("Berlin").score(_outcome("berlin")).passed


def test_contains_passes_when_the_needle_is_present() -> None:
    assert Contains("Berlin").score(_outcome("Your flight to Berlin is booked")).passed


def test_contains_is_case_insensitive_by_default() -> None:
    assert Contains("berlin").score(_outcome("Flight to BERLIN")).passed


def test_contains_requires_every_needle_and_names_the_missing_ones() -> None:
    score = Contains("Berlin", "flight", "€").score(_outcome("Berlin flight"))
    assert not score.passed
    assert "€" in score.detail
    assert score.value == pytest.approx(2 / 3), "partial credit reflects how close it got"


def test_contains_clips_a_huge_output_in_its_detail() -> None:
    score = Contains("nope").score(_outcome("x" * 5000))
    assert not score.passed
    assert len(score.detail) < 400


def test_contains_needs_at_least_one_needle() -> None:
    with pytest.raises(ScorerError, match="at least one"):
        Contains()


async def test_predicate_scores_with_a_sync_function() -> None:
    scorer = Predicate(lambda o: len(o.output) < 20, name="terse")
    assert (await scorer.score(_outcome("short"))).passed
    assert not (await scorer.score(_outcome("x" * 100))).passed


async def test_predicate_scores_with_an_async_function() -> None:
    async def check(outcome: RunOutcome) -> bool:
        return "ok" in outcome.output

    scorer = Predicate(check, name="says_ok")
    assert (await scorer.score(_outcome("all ok here"))).passed
    assert not (await scorer.score(_outcome("nope"))).passed


async def test_predicate_may_return_a_full_score() -> None:
    scorer = Predicate(
        lambda o: Score(scorer="judge", passed=False, value=0.4, detail="only partly right"),
        name="judge",
    )
    score = await scorer.score(_outcome("meh"))
    assert not score.passed and score.value == 0.4
    assert score.detail == "only partly right"


async def test_a_predicate_that_raises_is_a_broken_test_not_a_failing_case() -> None:
    """Recording it as a failure would hide a broken assertion behind a red case."""

    def boom(outcome: RunOutcome) -> bool:
        raise KeyError("typo in the scorer")

    with pytest.raises(ScorerError) as exc:
        await Predicate(boom, name="oops").score(_outcome("anything"))
    assert "oops" in str(exc.value)
    assert "fix the scorer" in str(exc.value)


async def test_a_predicate_returning_a_non_bool_is_rejected() -> None:
    with pytest.raises(ScorerError, match="must return a bool"):
        await Predicate(lambda o: "yes", name="stringly").score(_outcome(""))  # type: ignore[arg-type,return-value]


def test_predicate_rejects_a_non_callable() -> None:
    with pytest.raises(ScorerError, match="callable"):
        Predicate("not a function")  # type: ignore[arg-type]


def test_tool_called_passes_when_the_tool_ran() -> None:
    assert ToolCalled("refund").score(_outcome(tool_calls=(_call("refund", amount=10),))).passed


def test_tool_called_fails_and_lists_what_actually_ran() -> None:
    score = ToolCalled("refund").score(_outcome(tool_calls=(_call("lookup", order="a"),)))
    assert not score.passed
    assert "never called" in score.detail and "lookup" in score.detail


def test_tool_called_names_no_tools_when_the_run_called_none() -> None:
    score = ToolCalled("refund").score(_outcome("just prose"))
    assert not score.passed and "<no tools>" in score.detail


# ---------------------------------------------------------------------------
# 2. The trajectory assertion catches a WRONG ARGUMENT
# ---------------------------------------------------------------------------


def test_tool_trajectory_catches_a_wrong_argument() -> None:
    """The case this scorer exists for.

    An agent that answers "refunded!" while having called refund(amount=1000) on a $10
    order is a catastrophe no final-answer scorer can see.
    """
    outcome = _outcome("Refunded, all done!", tool_calls=(_call("refund", amount=1000),))

    assert Contains("Refunded").score(outcome).passed, "the prose looks fine..."

    score = ToolCalled("refund", {"amount": 10}).score(outcome)
    assert not score.passed, "...but the trajectory is wrong and must be caught"
    assert "{'amount': 10}" in score.detail
    assert "{'amount': 1000}" in score.detail, "the report must name the actual argument"


def test_tool_trajectory_passes_on_the_right_argument() -> None:
    outcome = _outcome("done", tool_calls=(_call("refund", amount=10),))
    assert ToolCalled("refund", {"amount": 10}).score(outcome).passed


def test_tool_arguments_are_matched_as_a_subset_by_default() -> None:
    """A test pins what it cares about and stays green when a tool grows an option."""
    outcome = _outcome(tool_calls=(_call("refund", amount=10, reason="damaged"),))
    assert ToolCalled("refund", {"amount": 10}).score(outcome).passed


def test_exact_argument_matching_is_available() -> None:
    outcome = _outcome(tool_calls=(_call("refund", amount=10, reason="damaged"),))
    assert not ToolCalled("refund", {"amount": 10}, exact=True).score(outcome).passed
    assert (
        ToolCalled("refund", {"amount": 10, "reason": "damaged"}, exact=True).score(outcome).passed
    )


def test_tool_called_can_pin_how_many_times() -> None:
    twice = _outcome(tool_calls=(_call("lookup", order="a"), _call("lookup", order="b")))
    assert ToolCalled("lookup", times=2).score(twice).passed
    score = ToolCalled("lookup", times=1).score(twice)
    assert not score.passed and "got 2" in score.detail


def test_tool_called_finds_the_right_call_among_several() -> None:
    outcome = _outcome(
        tool_calls=(
            _call("refund", amount=1),
            _call("refund", amount=10),
            _call("refund", amount=5),
        )
    )
    assert ToolCalled("refund", {"amount": 10}).score(outcome).passed


def test_tools_called_in_order_checks_a_subsequence() -> None:
    outcome = _outcome(
        tool_calls=(_call("lookup", order="a"), _call("audit"), _call("refund", amount=10))
    )
    assert ToolsCalledInOrder("lookup", "refund").score(outcome).passed, (
        "an extra step between two required ones must not fail the assertion"
    )


def test_tools_called_in_the_wrong_order_fails_and_says_what_was_missed() -> None:
    outcome = _outcome(tool_calls=(_call("refund", amount=10), _call("lookup", order="a")))
    score = ToolsCalledInOrder("lookup", "refund").score(outcome)
    assert not score.passed
    assert "never reached ['refund']" in score.detail


def test_tools_called_in_order_needs_at_least_one_tool() -> None:
    with pytest.raises(ScorerError, match="at least one"):
        ToolsCalledInOrder()


def test_tool_called_rejects_a_blank_name() -> None:
    with pytest.raises(ScorerError):
        ToolCalled("")


def test_run_outcome_exposes_the_trajectory_conveniently() -> None:
    outcome = _outcome(tool_calls=(_call("a"), _call("b"), _call("a", x=1)))
    assert outcome.tool_names == ("a", "b", "a")
    assert len(outcome.calls_of("a")) == 2


# ---------------------------------------------------------------------------
# 3. Running a suite against an Agent
# ---------------------------------------------------------------------------


async def test_a_suite_runs_every_case_and_reports_pass_fail() -> None:
    agent = _agent(fake_completion("391"), fake_completion("wrong"))
    suite = EvalSuite(
        "math",
        [
            EvalCase("right", "17 * 23?", scorers=[ExactMatch("391")]),
            EvalCase("wrong", "2 + 2?", scorers=[ExactMatch("4")]),
        ],
    )
    report = await suite.run(agent)

    assert report.total == 2
    assert report.passed == 1 and report.failed == 1
    assert report.pass_rate == 0.5
    assert not report.ok
    assert [r.case for r in report.failures] == ["wrong"]


async def test_a_case_needs_every_scorer_to_pass() -> None:
    """A case is an assertion, not an average."""
    agent = _agent(fake_completion("Berlin"))
    suite = EvalSuite(
        "s", [EvalCase("c", "where?", scorers=[Contains("Berlin"), Contains("Munich")])]
    )
    report = await suite.run(agent)
    assert not report.ok
    result = report.case("c")
    assert result is not None
    assert len(result.scores) == 2
    assert [s.passed for s in result.scores] == [True, False]


async def test_a_suite_scores_the_tool_trajectory_end_to_end() -> None:
    """The wrong-argument catch, through a real agent run rather than a hand-built outcome."""
    agent = _agent(
        fake_tool_call_completion("refund", {"amount": 1000}, call_id="t1"),
        fake_completion("Refunded, all done!"),
        tools=_registry(),
    )
    suite = EvalSuite(
        "refunds",
        [
            EvalCase(
                "ten-dollar-order",
                "refund order 42, it was $10",
                scorers=[Contains("Refunded"), ToolCalled("refund", {"amount": 10})],
            )
        ],
    )
    report = await suite.run(agent)

    assert not report.ok, "the answer reads fine but the agent refunded 100x too much"
    result = report.case("ten-dollar-order")
    assert result is not None
    assert [s.scorer for s in result.failures] == ["tool_called[refund]"]
    assert "1000" in result.failures[0].detail


async def test_each_case_starts_from_a_clean_conversation() -> None:
    """Otherwise case 2 answers in the context of case 1, and it is not a set of cases."""
    provider = FakeLLMProvider([fake_completion("a"), fake_completion("b")])
    agent = Agent(llm=LLM(provider=provider, model="fake", tracing=False))
    suite = EvalSuite(
        "s",
        [
            EvalCase("one", "first", scorers=[Contains("a")]),
            EvalCase("two", "second", scorers=[Contains("b")]),
        ],
    )
    await suite.run(agent)
    for seen in provider.calls:
        assert len([m for m in seen if m.role == "user"]) == 1


async def test_a_case_whose_run_raises_is_a_failing_case_not_an_aborted_suite() -> None:
    class Boom(FakeLLMProvider):
        async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kw):  # type: ignore[no-untyped-def]
            if any("explode" in m.content for m in messages):
                raise RuntimeError("provider exploded")
            return fake_completion("fine")

    agent = Agent(llm=LLM(provider=Boom(), model="fake", tracing=False))
    suite = EvalSuite(
        "s",
        [
            EvalCase("bad", "please explode", scorers=[Contains("anything")]),
            EvalCase("good", "behave", scorers=[Contains("fine")]),
        ],
    )
    report = await suite.run(agent)
    assert report.passed == 1 and report.failed == 1
    bad = report.case("bad")
    assert bad is not None and bad.error is not None
    assert "provider exploded" in bad.error
    assert "raised" in report.summary()


async def test_cases_can_run_concurrently() -> None:
    agent = _agent(*[fake_completion("ok") for _ in range(4)])
    suite = EvalSuite(
        "s",
        [EvalCase(f"c{i}", f"q{i}", scorers=[Contains("ok")]) for i in range(4)],
        concurrency=4,
    )
    report = await suite.run(agent)
    assert report.ok and report.total == 4


async def test_a_suite_can_evaluate_a_state_graph() -> None:
    """One suite, both halves of the framework."""

    class State(BaseModel):
        question: str
        answer: str = ""

    async def answer(state: State) -> dict[str, Any]:
        return {"answer": f"the answer to {state.question} is 42"}

    graph = StateGraph(State)
    graph.add_node("answer", answer)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    compiled = graph.compile()

    suite = EvalSuite(
        "graph",
        [
            EvalCase(
                "life",
                State(question="life"),
                scorers=[
                    Contains("42"),
                    Predicate(
                        lambda o: isinstance(o.state, State) and o.state.answer.endswith("42"),
                        name="state_field",
                    ),
                ],
            )
        ],
    )
    report = await suite.run(compiled)
    assert report.ok, report.summary()


async def test_a_graph_suite_rejects_a_string_input() -> None:
    class State(BaseModel):
        q: str = ""

    async def noop(state: State) -> None:
        return None

    graph = StateGraph(State)
    graph.add_node("n", noop)
    graph.set_entry_point("n")
    graph.add_edge("n", END)

    suite = EvalSuite("g", [EvalCase("c", "a string", scorers=[Contains("x")])])
    with pytest.raises(ScorerError, match="State"):
        await suite.run(graph.compile())


async def test_an_agent_suite_rejects_a_model_input() -> None:
    class State(BaseModel):
        q: str = ""

    suite = EvalSuite("a", [EvalCase("c", State(), scorers=[Contains("x")])])
    with pytest.raises(ScorerError, match="prompt string"):
        await suite.run(_agent(fake_completion("x")))


async def test_running_against_something_that_is_neither_is_a_type_error() -> None:
    suite = EvalSuite("s", [EvalCase("c", "q", scorers=[Contains("x")])])
    with pytest.raises(TypeError, match="Agent or a CompiledGraph"):
        await suite.run("not a target")  # type: ignore[arg-type]


def test_a_case_with_no_scorers_is_rejected() -> None:
    with pytest.raises(ScorerError, match="no scorers"):
        EvalCase("c", "q")


def test_a_case_with_a_non_scorer_is_rejected() -> None:
    with pytest.raises(ScorerError, match="Predicate"):
        EvalCase("c", "q", scorers=[object()])  # type: ignore[list-item]


def test_a_suite_with_no_cases_is_rejected() -> None:
    with pytest.raises(ScorerError, match="no cases"):
        EvalSuite("empty", [])


def test_duplicate_case_names_are_rejected() -> None:
    """compare() pairs cases by name; duplicates would silently pair the wrong two."""
    with pytest.raises(ScorerError, match="duplicate case names"):
        EvalSuite(
            "s",
            [
                EvalCase("same", "a", scorers=[Contains("x")]),
                EvalCase("same", "b", scorers=[Contains("x")]),
            ],
        )


def test_bad_concurrency_is_rejected() -> None:
    with pytest.raises(ScorerError, match="concurrency"):
        EvalSuite("s", [EvalCase("c", "q", scorers=[Contains("x")])], concurrency=0)


def test_eval_errors_are_catchable_as_actants_errors() -> None:
    with pytest.raises(ActantsError):
        EvalSuite("empty", [])


# ---------------------------------------------------------------------------
# 4. Cost and latency deltas between two runs
# ---------------------------------------------------------------------------


def _priced(content: str, model: str, prompt: int, completion: int) -> CompletionResult:
    """A completion priced through the real cost machinery, as a provider would report it."""
    from actants.cost.pricing import estimate_cost

    return CompletionResult(
        content=content,
        model=model,
        provider="openai",
        usage=TokenUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
        cost_usd=estimate_cost("openai", model, prompt, completion),
    )


async def test_cost_and_latency_delta_between_two_runs() -> None:
    """The number that answers 'should I switch models'."""
    expensive = Agent(
        llm=LLM(
            provider=FakeLLMProvider([_priced("391", "gpt-4o", 1000, 500)]),
            model="gpt-4o",
            tracing=False,
        )
    )
    cheap = Agent(
        llm=LLM(
            provider=FakeLLMProvider([_priced("391", "gpt-4o-mini", 1000, 500)]),
            model="gpt-4o-mini",
            tracing=False,
        )
    )
    suite = EvalSuite("math", [EvalCase("mult", "17 * 23?", scorers=[ExactMatch("391")])])

    baseline = await suite.run(expensive)
    candidate = await suite.run(cheap)

    # gpt-4o: 1000 in @ $2.50/1M + 500 out @ $10/1M = $0.0075
    # gpt-4o-mini: 1000 in @ $0.15/1M + 500 out @ $0.60/1M = $0.00045
    assert baseline.total_cost_usd == pytest.approx(0.0075)
    assert candidate.total_cost_usd == pytest.approx(0.00045)

    delta = candidate.compare(baseline)
    assert delta.cost_delta_usd == pytest.approx(-0.00705), "the cheap model saves money"
    assert delta.cost_change_pct == pytest.approx(-94.0)
    assert not delta.regressed, "and breaks nothing"
    assert delta.pass_rate_delta == 0.0
    assert delta.latency_delta_ms == pytest.approx(
        candidate.total_latency_ms - baseline.total_latency_ms
    )

    summary = delta.summary()
    assert "cost:" in summary and "latency:" in summary
    assert "no correctness change" in summary


async def test_a_delta_names_the_cases_that_regressed() -> None:
    """A cheaper model that saves 94% and breaks two cases is a decision, not a win."""
    good = _agent(fake_completion("391"), fake_completion("Berlin"))
    bad = _agent(fake_completion("391"), fake_completion("I don't know"))
    suite = EvalSuite(
        "mixed",
        [
            EvalCase("math", "17 * 23?", scorers=[ExactMatch("391")]),
            EvalCase("geo", "capital of Germany?", scorers=[Contains("Berlin")]),
        ],
    )

    baseline = await suite.run(good)
    candidate = await suite.run(bad)
    delta = candidate.compare(baseline)

    assert delta.regressed
    assert delta.regressions == ("geo",)
    assert delta.fixes == ()
    assert delta.pass_rate_delta == pytest.approx(-0.5)
    assert "REGRESSED (1): geo" in delta.summary()


async def test_a_delta_names_the_cases_a_change_fixed() -> None:
    bad = _agent(fake_completion("no idea"))
    good = _agent(fake_completion("Berlin"))
    suite = EvalSuite("geo", [EvalCase("capital", "capital?", scorers=[Contains("Berlin")])])

    baseline = await suite.run(bad)
    candidate = await suite.run(good)
    delta = candidate.compare(baseline)

    assert delta.fixes == ("capital",)
    assert not delta.regressed
    assert "fixed (1): capital" in delta.summary()


async def test_a_delta_reports_cases_added_and_dropped() -> None:
    agent_a = _agent(fake_completion("x"))
    agent_b = _agent(fake_completion("x"), fake_completion("x"))
    old = EvalSuite("s", [EvalCase("kept", "q", scorers=[Contains("x")])])
    new = EvalSuite(
        "s",
        [
            EvalCase("kept", "q", scorers=[Contains("x")]),
            EvalCase("added", "q2", scorers=[Contains("x")]),
        ],
    )
    baseline = await old.run(agent_a)
    candidate = await new.run(agent_b)
    delta = candidate.compare(baseline)
    assert delta.only_in_candidate == ("added",)
    assert delta.only_in_baseline == ()
    assert "new cases: added" in delta.summary()


async def test_a_free_baseline_reports_no_percentage_rather_than_a_fake_one() -> None:
    """Local Ollama is genuinely free; '+0%' for free-to-$4 would be a lie."""
    local = _agent(fake_completion("ok"))
    hosted = Agent(
        llm=LLM(
            provider=FakeLLMProvider([_priced("ok", "gpt-4o", 1000, 100)]),
            model="gpt-4o",
            tracing=False,
        )
    )
    suite = EvalSuite("s", [EvalCase("c", "q", scorers=[Contains("ok")])])
    baseline = await suite.run(local)
    candidate = await suite.run(hosted)
    delta = candidate.compare(baseline)

    assert delta.baseline_cost_usd == 0.0
    assert delta.cost_change_pct is None
    assert "%" not in delta.summary().splitlines()[0]


async def test_a_report_sums_cost_over_multi_step_runs() -> None:
    """A tool-calling run pays for every step, and the report must say so."""
    agent = Agent(
        llm=LLM(
            provider=FakeLLMProvider(
                [
                    CompletionResult(
                        content="",
                        model="gpt-4o",
                        provider="openai",
                        cost_usd=0.01,
                        tool_calls=[ToolCall(id="t1", name="lookup", arguments={"order": "42"})],
                    ),
                    CompletionResult(
                        content="found it", model="gpt-4o", provider="openai", cost_usd=0.02
                    ),
                ]
            ),
            model="gpt-4o",
            tracing=False,
        ),
        tools=_registry(),
    )
    suite = EvalSuite("s", [EvalCase("c", "find order 42", scorers=[Contains("found")])])
    report = await suite.run(agent)
    assert report.total_cost_usd == pytest.approx(0.03)
    result = report.case("c")
    assert result is not None and result.outcome is not None
    assert result.outcome.tool_names == ("lookup",)


def test_compare_rejects_something_that_is_not_a_report() -> None:
    with pytest.raises(TypeError, match="EvalReport"):
        EvalReport(suite="s").compare({"passed": 1})  # type: ignore[arg-type]


def test_an_empty_report_has_a_zero_pass_rate_not_a_zero_division() -> None:
    report = EvalReport(suite="s")
    assert report.pass_rate == 0.0
    assert report.ok, "vacuously true: nothing failed"


# ---------------------------------------------------------------------------
# 5. The report is readable and machine-readable
# ---------------------------------------------------------------------------


async def test_a_passing_report_summary_is_one_line() -> None:
    agent = _agent(fake_completion("391"))
    suite = EvalSuite("math", [EvalCase("mult", "17 * 23?", scorers=[ExactMatch("391")])])
    summary = (await suite.run(agent)).summary()
    assert summary.count("\n") == 0
    assert "1/1 passed" in summary and "100%" in summary


async def test_a_failing_report_summary_names_every_failure_and_why() -> None:
    agent = _agent(fake_completion("wrong"))
    suite = EvalSuite("math", [EvalCase("mult", "17 * 23?", scorers=[ExactMatch("391")])])
    summary = (await suite.run(agent)).summary()
    assert "0/1 passed" in summary
    assert "FAIL mult" in summary
    assert "exact_match" in summary
    assert "'391'" in summary


async def test_a_report_serializes_to_json_for_ci() -> None:
    import json

    agent = _agent(
        fake_tool_call_completion("lookup", {"order": "42"}, call_id="t1"),
        fake_completion("order 42 found"),
        tools=_registry(),
    )
    suite = EvalSuite(
        "orders",
        [
            EvalCase(
                "find",
                "find order 42",
                scorers=[Contains("found"), ToolCalled("lookup", {"order": "42"})],
                tags=("smoke",),
            )
        ],
    )
    report = await suite.run(agent)

    payload = json.loads(report.to_json())
    assert payload["suite"] == "orders"
    assert payload["passed"] == 1 and payload["total"] == 1
    assert payload["pass_rate"] == 1.0
    case = payload["cases"][0]
    assert case["name"] == "find"
    assert case["tags"] == ["smoke"]
    assert case["tool_calls"] == [{"name": "lookup", "arguments": {"order": "42"}}]
    assert {s["scorer"] for s in case["scores"]} == {"contains", "tool_called[lookup]"}


async def test_a_report_records_the_output_of_each_case() -> None:
    agent = _agent(fake_completion("the answer"))
    suite = EvalSuite("s", [EvalCase("c", "q", scorers=[Contains("answer")])])
    report = await suite.run(agent)
    assert report.to_dict()["cases"][0]["output"] == "the answer"


async def test_case_lookup_returns_none_for_an_unknown_name() -> None:
    agent = _agent(fake_completion("x"))
    suite = EvalSuite("s", [EvalCase("c", "q", scorers=[Contains("x")])])
    report = await suite.run(agent)
    assert report.case("nope") is None


def test_suite_repr_and_len() -> None:
    suite = EvalSuite(
        "s",
        [
            EvalCase("a", "q", scorers=[Contains("x")]),
            EvalCase("b", "q", scorers=[Contains("x")]),
        ],
    )
    assert len(suite) == 2
    assert "cases=2" in repr(suite)
