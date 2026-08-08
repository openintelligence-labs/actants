"""Score an agent or a graph against a set of cases, and diff two runs.

An agent's behaviour is a function of model, prompt, and tools. This is the half of the
answer that says *whether the change was good*: run the same cases against the old and
the new, and get back per-case pass/fail plus what the swap cost in dollars and
milliseconds.

Scoring an agent means scoring two things, and the second is the one most harnesses
miss. The final answer matters, but so does the **trajectory** — which tools the model
reached for and with what arguments. A run that produces the right answer by calling
``refund(amount=1000)`` instead of ``refund(amount=10)`` is not a passing run.

Example::

    suite = EvalSuite(
        name="booking",
        cases=[
            EvalCase("berlin", "book a flight to Berlin",
                     scorers=[Contains("Berlin"),
                              ToolCalled("search_flights", {"city": "Berlin"})]),
            EvalCase("math", "what is 17 * 23?", scorers=[ExactMatch("391")]),
        ],
    )
    report = await suite.run(agent)
    print(report.summary())

    # Did the cheaper model break anything?
    baseline = await suite.run(agent_4o)
    candidate = await suite.run(agent_4o_mini)
    print(candidate.compare(baseline).summary())

Every scorer is a plain object with a ``score()`` method, so a bespoke one is a class with
one method — or, for the common case, :class:`Predicate` wrapping a lambda.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from actants.errors import ScorerError
from actants.llm.base import ToolCall

if TYPE_CHECKING:
    from actants.agents.agent import Agent
    from actants.graph.state_graph import CompiledGraph


@dataclass(frozen=True)
class RunOutcome:
    """What one case's run produced, in the vocabulary every scorer reads.

    Deliberately not an :class:`~actants.agents.agent.AgentResult` or a
    :class:`~actants.graph.state_graph.GraphResult`: a scorer written against this works
    on both, which is what lets one suite evaluate an agent today and the graph it grows
    into tomorrow.
    """

    #: The run's final text answer.
    output: str
    #: Every tool call the run made, in order — the trajectory.
    tool_calls: tuple[ToolCall, ...] = ()
    #: Cost as the provider reported it, summed over the run's LLM calls.
    cost_usd: float = 0.0
    #: Wall-clock duration of the whole run.
    latency_ms: float = 0.0
    #: Prompt + completion tokens over the run's LLM calls.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: The state a graph run finished with; None for an agent run.
    state: BaseModel | None = None

    def calls_of(self, name: str) -> tuple[ToolCall, ...]:
        """Every call to the tool named ``name``, in order."""
        return tuple(c for c in self.tool_calls if c.name == name)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The trajectory as bare names, for a quick order assertion."""
        return tuple(c.name for c in self.tool_calls)


@dataclass(frozen=True)
class Score:
    """One scorer's verdict on one case.

    ``passed`` is the answer; ``detail`` is what a human needs to fix it, and is the
    reason a failing eval report is actionable rather than a wall of red.
    """

    scorer: str
    passed: bool
    detail: str = ""
    #: For scorers that are not binary — an LLM judge, a similarity threshold. Binary
    #: scorers leave it at 1.0/0.0 so aggregate scoring works uniformly.
    value: float = 0.0


@runtime_checkable
class Scorer(Protocol):
    """What an eval case checks. Implement ``name`` and ``score``.

    ``score`` may be sync or async — an LLM judge needs to await, an exact-match does
    not, and forcing the second to be async would be ceremony.
    """

    @property
    def name(self) -> str:
        """How this scorer is identified in the report."""
        ...

    def score(self, outcome: RunOutcome) -> Score | Awaitable[Score]:
        """Judge one run."""
        ...


class ExactMatch:
    """The final answer must equal ``expected``.

    ``strip`` and ``case_sensitive`` default to the forgiving reading, because a model
    that answers ``"391\\n"`` has not got the answer wrong.
    """

    def __init__(self, expected: str, *, strip: bool = True, case_sensitive: bool = True) -> None:
        self.expected = expected
        self.strip = strip
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "exact_match"

    def score(self, outcome: RunOutcome) -> Score:
        got, want = _normalize(outcome.output, self.expected, self.strip, self.case_sensitive)
        passed = got == want
        return Score(
            scorer=self.name,
            passed=passed,
            value=1.0 if passed else 0.0,
            detail="" if passed else f"expected {want!r}, got {got!r}",
        )


class Contains:
    """The final answer must contain ``needle`` (all of them, if given several)."""

    def __init__(self, *needles: str, case_sensitive: bool = False) -> None:
        if not needles:
            raise ScorerError(
                "Contains() needs at least one string to look for. "
                "Example: Contains('Berlin'), or Contains('Berlin', 'flight')."
            )
        self.needles = needles
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "contains"

    def score(self, outcome: RunOutcome) -> Score:
        haystack = outcome.output if self.case_sensitive else outcome.output.lower()
        missing = [
            n for n in self.needles if (n if self.case_sensitive else n.lower()) not in haystack
        ]
        passed = not missing
        return Score(
            scorer=self.name,
            passed=passed,
            value=(len(self.needles) - len(missing)) / len(self.needles),
            detail="" if passed else f"missing {missing} in output {_clip(outcome.output)}",
        )


class Predicate:
    """Score with a caller-supplied function. Sync or async, returning ``bool`` or ``Score``.

    The escape hatch that keeps the built-in scorers from having to cover everything::

        Predicate(lambda o: len(o.output) < 200, name="terse")
        Predicate(check_against_db, name="row_written")     # async is fine
    """

    def __init__(
        self,
        fn: Callable[[RunOutcome], bool | Score | Awaitable[bool | Score]],
        *,
        name: str = "predicate",
        detail: str = "",
    ) -> None:
        if not callable(fn):
            raise ScorerError(
                f"Predicate() needs a callable taking a RunOutcome, got "
                f"{type(fn).__name__!r}. Example: "
                "Predicate(lambda o: 'Berlin' in o.output, name='mentions_berlin')."
            )
        self._fn = fn
        self._name = name
        self._detail = detail

    @property
    def name(self) -> str:
        return self._name

    async def score(self, outcome: RunOutcome) -> Score:
        try:
            verdict = self._fn(outcome)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        except Exception as exc:
            # A scorer that raises is a bug in the test, not a failing case. Recording it
            # as a failure would hide a broken assertion behind a red case.
            raise ScorerError(
                f"Predicate scorer {self._name!r} raised {type(exc).__name__}: {exc}. "
                "A scorer must return True/False or a Score; fix the scorer rather than "
                "the agent."
            ) from exc
        if isinstance(verdict, Score):
            return verdict
        if not isinstance(verdict, bool):
            raise ScorerError(
                f"Predicate scorer {self._name!r} returned {type(verdict).__name__!r}; "
                "it must return a bool or a Score."
            )
        return Score(
            scorer=self._name,
            passed=verdict,
            value=1.0 if verdict else 0.0,
            detail="" if verdict else (self._detail or f"{self._name} returned False"),
        )


class ToolCalled:
    """The run must have called ``tool`` — optionally with exactly these arguments.

    The trajectory assertion. For an agent, *which* tool ran with *what* arguments is
    often more load-bearing than the prose it wrote afterwards: a refund agent that
    answers "done!" while having called ``refund(amount=1000)`` on a $10 order is a
    catastrophe the final answer cannot see.

    ``arguments`` is checked as a **subset** by default, so a test pins the arguments it
    cares about and stays green when a tool grows an optional one. Pass ``exact=True`` to
    require the argument dict to match exactly.
    """

    def __init__(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        exact: bool = False,
        times: int | None = None,
    ) -> None:
        if not tool or not isinstance(tool, str):
            raise ScorerError(f"ToolCalled() needs a tool name, got {tool!r}.")
        if times is not None and times < 0:
            raise ScorerError(f"ToolCalled(times=...) must be >= 0, got {times}.")
        self.tool = tool
        self.arguments = arguments
        self.exact = exact
        self.times = times

    @property
    def name(self) -> str:
        return f"tool_called[{self.tool}]"

    def score(self, outcome: RunOutcome) -> Score:
        calls = outcome.calls_of(self.tool)
        if not calls:
            return Score(
                scorer=self.name,
                passed=False,
                detail=(
                    f"tool {self.tool!r} was never called; the run called "
                    f"{list(outcome.tool_names) or '<no tools>'}"
                ),
            )
        if self.arguments is None:
            return self._count_verdict(calls, matching=calls)

        matching = [c for c in calls if self._args_match(c.arguments)]
        if not matching:
            # Naming the closest actual call is the difference between "assertion failed"
            # and "you passed the wrong city" — the wrong-argument case is the one this
            # scorer exists for.
            return Score(
                scorer=self.name,
                passed=False,
                detail=(
                    f"tool {self.tool!r} was called {len(calls)} time(s), but never with "
                    f"{'exactly ' if self.exact else ''}{self.arguments!r}. "
                    f"Actual: {[dict(c.arguments) for c in calls]}"
                ),
            )
        return self._count_verdict(calls, matching=matching)

    def _count_verdict(self, calls: Sequence[ToolCall], *, matching: Sequence[ToolCall]) -> Score:
        if self.times is not None and len(matching) != self.times:
            return Score(
                scorer=self.name,
                passed=False,
                detail=(
                    f"expected {self.tool!r} to match {self.times} time(s), got {len(matching)}"
                ),
            )
        return Score(scorer=self.name, passed=True, value=1.0)

    def _args_match(self, actual: dict[str, Any]) -> bool:
        assert self.arguments is not None
        if self.exact:
            return actual == self.arguments
        return all(actual.get(k) == v for k, v in self.arguments.items())


class ToolsCalledInOrder:
    """The run's tool trajectory must contain ``tools`` as a subsequence.

    A subsequence, not an exact list: a model that inserts an extra lookup between two
    required steps has still done the required steps in the required order, and pinning
    the exact list makes a test that fails on every harmless improvement.
    """

    def __init__(self, *tools: str) -> None:
        if not tools:
            raise ScorerError(
                "ToolsCalledInOrder() needs at least one tool name. "
                "Example: ToolsCalledInOrder('search', 'book')."
            )
        self.tools = tools

    @property
    def name(self) -> str:
        return "tools_in_order"

    def score(self, outcome: RunOutcome) -> Score:
        remaining = list(self.tools)
        for called in outcome.tool_names:
            if remaining and called == remaining[0]:
                remaining.pop(0)
        passed = not remaining
        return Score(
            scorer=self.name,
            passed=passed,
            value=(len(self.tools) - len(remaining)) / len(self.tools),
            detail=(
                ""
                if passed
                else (
                    f"expected {list(self.tools)} in order; the run called "
                    f"{list(outcome.tool_names)} and never reached {remaining}"
                )
            ),
        )


@dataclass
class EvalCase:
    """One input, and what a good answer to it looks like.

    ``input`` is the prompt for an agent, or the state for a graph. ``scorers`` all have
    to pass for the case to pass — a case is an assertion, not an average.
    """

    name: str
    input: str | BaseModel
    scorers: Sequence[Scorer] = field(default_factory=tuple)
    #: Free-form labels, carried into the report so a CI job can slice by them.
    tags: tuple[str, ...] = ()
    #: Overrides the suite's step budget for this one case.
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ScorerError("Every EvalCase needs a name; it is how the report identifies it.")
        if not self.scorers:
            raise ScorerError(
                f"EvalCase {self.name!r} has no scorers, so it can neither pass nor fail. "
                "Add at least one, e.g. scorers=[Contains('Berlin')]."
            )
        for scorer in self.scorers:
            if not isinstance(scorer, Scorer):
                raise ScorerError(
                    f"EvalCase {self.name!r} was given {type(scorer).__name__!r} as a "
                    "scorer, which has no name/score. Use a built-in "
                    "(ExactMatch, Contains, ToolCalled) or wrap a function: "
                    "Predicate(lambda o: ..., name='...')."
                )


@dataclass
class CaseResult:
    """What one case produced: its scores, and what the run cost.

    ``error`` is set when the run itself raised — a crashed case is a failing case, and
    the traceback is preserved as a string so a report survives being written to JSON.
    """

    case: str
    passed: bool
    scores: list[Score] = field(default_factory=list)
    outcome: RunOutcome | None = None
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tags: tuple[str, ...] = ()

    @property
    def failures(self) -> list[Score]:
        """Only the scores that failed — what a report prints under a red case."""
        return [s for s in self.scores if not s.passed]


@dataclass(frozen=True)
class ReportDelta:
    """The difference between two :class:`EvalReport` runs.

    The number that actually answers "should I switch models": what the swap did to
    correctness, to spend, and to latency, all at once. A cheaper model that costs 40%
    less and fails two more cases is a decision, not an improvement, and this makes both
    halves visible in one object.
    """

    #: Cases that passed in the baseline and fail now. The blocker list.
    regressions: tuple[str, ...]
    #: Cases that failed in the baseline and pass now.
    fixes: tuple[str, ...]
    #: Cases present in one report and not the other.
    only_in_candidate: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    #: candidate - baseline. Negative is cheaper / faster.
    cost_delta_usd: float
    latency_delta_ms: float
    pass_rate_delta: float
    baseline_cost_usd: float
    candidate_cost_usd: float
    baseline_latency_ms: float
    candidate_latency_ms: float

    @property
    def regressed(self) -> bool:
        """True when any case that used to pass now fails."""
        return bool(self.regressions)

    @property
    def cost_change_pct(self) -> float | None:
        """Relative cost change, or None when the baseline was free (e.g. local Ollama).

        None rather than 0.0 or infinity: dividing by a zero baseline is undefined, and a
        report that prints "+0%" for a run that went from free to $4 is worse than one
        that admits the ratio is meaningless.
        """
        if self.baseline_cost_usd == 0.0:
            return None
        return (self.cost_delta_usd / self.baseline_cost_usd) * 100

    def summary(self) -> str:
        """A human-readable diff, for a terminal or a PR comment."""
        lines = [
            f"cost:    {self.baseline_cost_usd:.6f} -> {self.candidate_cost_usd:.6f} USD "
            f"({self.cost_delta_usd:+.6f})",
            f"latency: {self.baseline_latency_ms:.0f} -> {self.candidate_latency_ms:.0f} ms "
            f"({self.latency_delta_ms:+.0f})",
            f"pass rate: {self.pass_rate_delta:+.1%}",
        ]
        pct = self.cost_change_pct
        if pct is not None:
            lines[0] += f"  [{pct:+.1f}%]"
        if self.regressions:
            lines.append(f"REGRESSED ({len(self.regressions)}): {', '.join(self.regressions)}")
        if self.fixes:
            lines.append(f"fixed ({len(self.fixes)}): {', '.join(self.fixes)}")
        if self.only_in_candidate:
            lines.append(f"new cases: {', '.join(self.only_in_candidate)}")
        if self.only_in_baseline:
            lines.append(f"dropped cases: {', '.join(self.only_in_baseline)}")
        if not self.regressions and not self.fixes:
            lines.append("no correctness change")
        return "\n".join(lines)


@dataclass
class EvalReport:
    """The result of running a suite: per-case verdicts plus the aggregates.

    :meth:`summary` is for a human, :meth:`to_dict` for CI. :meth:`compare` against an
    earlier report is what turns this from a test result into a decision.
    """

    suite: str
    results: list[CaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    #: What the whole suite spent and took, summed over its cases.
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed. 0.0 for an empty suite, not a ZeroDivisionError."""
        return self.passed / self.total if self.results else 0.0

    @property
    def ok(self) -> bool:
        """True when every case passed. What a CI job exits on."""
        return self.failed == 0

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def case(self, name: str) -> CaseResult | None:
        """Look one case up by name."""
        return next((r for r in self.results if r.case == name), None)

    def summary(self) -> str:
        """A readable report: the headline, then every failure with its reason."""
        head = (
            f"{self.suite}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%})  "
            f"${self.total_cost_usd:.6f}  {self.total_latency_ms:.0f}ms"
        )
        if self.ok:
            return head
        lines = [head, ""]
        for result in self.failures:
            if result.error is not None:
                lines.append(f"  FAIL {result.case}: raised {result.error}")
                continue
            reasons = "; ".join(f"{s.scorer}: {s.detail or 'failed'}" for s in result.failures)
            lines.append(f"  FAIL {result.case}: {reasons}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable form, for a CI artifact or a dashboard."""
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "cases": [
                {
                    "name": r.case,
                    "passed": r.passed,
                    "tags": list(r.tags),
                    "cost_usd": round(r.cost_usd, 6),
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "output": r.outcome.output if r.outcome is not None else None,
                    "tool_calls": (
                        [{"name": c.name, "arguments": c.arguments} for c in r.outcome.tool_calls]
                        if r.outcome is not None
                        else []
                    ),
                    "scores": [
                        {
                            "scorer": s.scorer,
                            "passed": s.passed,
                            "value": round(s.value, 4),
                            "detail": s.detail,
                        }
                        for s in r.scores
                    ],
                }
                for r in self.results
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """:meth:`to_dict` as a JSON string, for writing straight to a CI artifact."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def compare(self, baseline: EvalReport) -> ReportDelta:
        """Diff this report against an earlier one — old model vs new, before vs after.

        ``self`` is the candidate and ``baseline`` is what it is being judged against, so
        a negative ``cost_delta_usd`` means the candidate is cheaper.
        """
        if not isinstance(baseline, EvalReport):
            raise TypeError(
                f"compare() expects another EvalReport, got {type(baseline).__name__!r}. "
                "Run the suite twice — once per model — and compare the two reports."
            )
        mine = {r.case: r for r in self.results}
        theirs = {r.case: r for r in baseline.results}
        shared = mine.keys() & theirs.keys()
        return ReportDelta(
            regressions=tuple(sorted(c for c in shared if theirs[c].passed and not mine[c].passed)),
            fixes=tuple(sorted(c for c in shared if not theirs[c].passed and mine[c].passed)),
            only_in_candidate=tuple(sorted(mine.keys() - theirs.keys())),
            only_in_baseline=tuple(sorted(theirs.keys() - mine.keys())),
            cost_delta_usd=self.total_cost_usd - baseline.total_cost_usd,
            latency_delta_ms=self.total_latency_ms - baseline.total_latency_ms,
            pass_rate_delta=self.pass_rate - baseline.pass_rate,
            baseline_cost_usd=baseline.total_cost_usd,
            candidate_cost_usd=self.total_cost_usd,
            baseline_latency_ms=baseline.total_latency_ms,
            candidate_latency_ms=self.total_latency_ms,
        )


class EvalSuite:
    """A named set of cases, runnable against an Agent or a CompiledGraph.

    ``concurrency`` runs cases in parallel; the default of 1 is sequential, which is what
    a suite sharing one :class:`~actants.agents.agent.Agent` needs — every case is an
    independent question, so the agent's conversation is reset between them.

    Example::

        suite = EvalSuite("booking", [EvalCase("berlin", "book Berlin",
                                               scorers=[Contains("Berlin")])])
        report = await suite.run(agent)
        assert report.ok, report.summary()
    """

    def __init__(
        self,
        name: str,
        cases: Iterable[EvalCase],
        *,
        max_steps: int | None = None,
        concurrency: int = 1,
    ) -> None:
        self.name = name
        self.cases = list(cases)
        if not self.cases:
            raise ScorerError(
                f"EvalSuite {name!r} has no cases. Add at least one EvalCase, e.g. "
                "EvalCase('math', 'what is 2+2?', scorers=[Contains('4')])."
            )
        counts = Counter(c.name for c in self.cases)
        duplicates = sorted(name for name, n in counts.items() if n > 1)
        if duplicates:
            raise ScorerError(
                f"EvalSuite {name!r} has duplicate case names {duplicates}. Case names "
                "identify a case in the report and across runs, so they must be unique — "
                "compare() would otherwise silently pair the wrong two."
            )
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ScorerError(
                f"concurrency must be an integer >= 1, got {concurrency!r}. "
                "1 (the default) runs cases one at a time."
            )
        self.max_steps = max_steps
        self.concurrency = concurrency

    async def run(self, target: Agent | CompiledGraph[Any]) -> EvalReport:
        """Run every case against ``target`` and score the results.

        Accepts an :class:`~actants.agents.agent.Agent` or a
        :class:`~actants.graph.state_graph.CompiledGraph`; the difference is confined to
        how one case is executed, so the same suite scores both.
        """
        runner = _make_runner(target, self.max_steps)
        report = EvalReport(suite=self.name)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(case: EvalCase) -> CaseResult:
            async with semaphore:
                return await _run_case(case, runner)

        # gather rather than a loop even at concurrency=1: the semaphore is what enforces
        # the limit, so both paths go through identical code and cannot drift.
        results = await asyncio.gather(*(one(c) for c in self.cases))
        report.results = list(results)
        report.total_cost_usd = sum(r.cost_usd for r in results)
        report.total_latency_ms = sum(r.latency_ms for r in results)
        return report

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return f"EvalSuite(name={self.name!r}, cases={len(self.cases)})"


#: How one case is executed. Returned by :func:`_make_runner`, which is the only place
#: that knows an Agent from a CompiledGraph.
type _Runner = Callable[[EvalCase], Awaitable[RunOutcome]]


def _make_runner(target: Agent | CompiledGraph[Any], suite_max_steps: int | None) -> _Runner:
    """Adapt an Agent or a CompiledGraph to the single shape the suite drives."""
    from actants.agents.agent import Agent as _Agent
    from actants.graph.state_graph import CompiledGraph as _CompiledGraph

    if isinstance(target, _Agent):

        async def run_agent(case: EvalCase) -> RunOutcome:
            if not isinstance(case.input, str):
                raise ScorerError(
                    f"Case {case.name!r} has a {type(case.input).__name__!r} input, but "
                    "the suite is running against an Agent, whose input is a prompt "
                    "string. Pass a str, or run the suite against a CompiledGraph."
                )
            # Each case is an independent question; without this, case 2 would answer in
            # the context of case 1 and the suite would not be a set of cases at all.
            target.reset()
            started = time.perf_counter()
            result = await target.run(case.input, max_steps=case.max_steps or suite_max_steps)
            elapsed = (time.perf_counter() - started) * 1000
            calls = tuple(c for step in result.steps for c in step.tool_calls)
            return RunOutcome(
                output=result.content,
                tool_calls=calls,
                cost_usd=sum(s.completion.cost_usd for s in result.steps),
                latency_ms=elapsed,
                prompt_tokens=sum(s.completion.usage.prompt_tokens for s in result.steps),
                completion_tokens=sum(s.completion.usage.completion_tokens for s in result.steps),
            )

        return run_agent

    if isinstance(target, _CompiledGraph):
        graph: CompiledGraph[Any] = target

        async def run_graph(case: EvalCase) -> RunOutcome:
            if not isinstance(case.input, BaseModel):
                raise ScorerError(
                    f"Case {case.name!r} has a {type(case.input).__name__!r} input, but "
                    f"the suite is running against a graph over "
                    f"{graph.state_type.__name__}. Pass a {graph.state_type.__name__} "
                    "instance as the case input."
                )
            started = time.perf_counter()
            result = await graph.invoke(case.input)
            elapsed = (time.perf_counter() - started) * 1000
            return RunOutcome(
                output=_graph_output(result.state),
                latency_ms=elapsed,
                state=result.state,
            )

        return run_graph

    raise TypeError(
        f"EvalSuite.run() expects an Agent or a CompiledGraph, got "
        f"{type(target).__name__!r}. Compile a StateGraph first: "
        "report = await suite.run(graph.compile())."
    )


def _graph_output(state: BaseModel) -> str:
    """Render a graph's final state as the text a text scorer reads.

    A graph has no single "answer" field, so the whole state is serialized. Scorers that
    care about one field read ``outcome.state`` directly, which is why it is carried.
    """
    return state.model_dump_json()


async def _run_case(case: EvalCase, runner: _Runner) -> CaseResult:
    """Execute one case and apply its scorers.

    A run that raises is a failing case with the exception recorded — one broken case must
    not abort a whole suite. A *scorer* that raises is a different thing entirely and is
    allowed to propagate: that is a bug in the test.
    """
    try:
        outcome = await runner(case)
    except ScorerError:
        raise
    except Exception as exc:
        return CaseResult(
            case=case.name,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            tags=case.tags,
        )

    scores: list[Score] = []
    for scorer in case.scorers:
        verdict = scorer.score(outcome)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        scores.append(verdict)

    return CaseResult(
        case=case.name,
        passed=all(s.passed for s in scores),
        scores=scores,
        outcome=outcome,
        cost_usd=outcome.cost_usd,
        latency_ms=outcome.latency_ms,
        tags=case.tags,
    )


def _normalize(got: str, want: str, strip: bool, case_sensitive: bool) -> tuple[str, str]:
    if strip:
        got, want = got.strip(), want.strip()
    if not case_sensitive:
        got, want = got.lower(), want.lower()
    return got, want


def _clip(text: str, limit: int = 200) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


__all__ = [
    "CaseResult",
    "Contains",
    "EvalCase",
    "EvalReport",
    "EvalSuite",
    "ExactMatch",
    "Predicate",
    "ReportDelta",
    "RunOutcome",
    "Score",
    "Scorer",
    "ToolCalled",
    "ToolsCalledInOrder",
]
