"""Test doubles, run recording/replay, and the evaluation harness.

Three things that answer one question — *did my change break the agent?*

* :mod:`~actants.testing.fakes` — scripted providers for a unit test.
* :mod:`~actants.testing.recording` — record a real run to JSONL, replay it offline.
* :mod:`~actants.testing.evals` — score runs against cases, and diff two runs' cost.
"""

from __future__ import annotations

from actants.testing.evals import (
    CaseResult,
    Contains,
    EvalCase,
    EvalReport,
    EvalSuite,
    ExactMatch,
    Predicate,
    ReportDelta,
    RunOutcome,
    Score,
    Scorer,
    ToolCalled,
    ToolsCalledInOrder,
)
from actants.testing.fakes import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    fake_completion,
    fake_tool_call_completion,
)
from actants.testing.recording import (
    FORMAT_VERSION,
    MatchMode,
    RecordedExchange,
    RecordedRequest,
    Recording,
    RecordingHeader,
    RecordingProvider,
    ReplayProvider,
    RunRecorder,
    iter_exchanges,
)

__all__ = [
    "FORMAT_VERSION",
    "CaseResult",
    "Contains",
    "EvalCase",
    "EvalReport",
    "EvalSuite",
    "ExactMatch",
    "FakeEmbeddingProvider",
    "FakeLLMProvider",
    "MatchMode",
    "Predicate",
    "RecordedExchange",
    "RecordedRequest",
    "Recording",
    "RecordingHeader",
    "RecordingProvider",
    "ReplayProvider",
    "ReportDelta",
    "RunOutcome",
    "RunRecorder",
    "Score",
    "Scorer",
    "ToolCalled",
    "ToolsCalledInOrder",
    "fake_completion",
    "fake_tool_call_completion",
    "iter_exchanges",
]
