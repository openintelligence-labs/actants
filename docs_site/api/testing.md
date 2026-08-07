# Testing

Fakes and helpers for testing apps that depend on actants without making
network calls.

::: actants.testing.fakes.FakeLLMProvider

::: actants.testing.fakes.FakeEmbeddingProvider

::: actants.testing.fakes.fake_completion

::: actants.testing.fakes.fake_tool_call_completion

## Recording and replay

Record a real run, then replay it with no network. See
[Testing agents](../concepts/testing.md).

::: actants.testing.recording.RunRecorder

::: actants.testing.recording.Recording

::: actants.testing.recording.ReplayProvider

::: actants.testing.recording.RecordingProvider

::: actants.testing.recording.RecordedExchange

::: actants.testing.recording.RecordingHeader

::: actants.testing.recording.iter_exchanges

## Evaluation

::: actants.testing.evals.EvalSuite

::: actants.testing.evals.EvalCase

::: actants.testing.evals.EvalReport

::: actants.testing.evals.ReportDelta

::: actants.testing.evals.Score

### Scorers

::: actants.testing.evals.ExactMatch

::: actants.testing.evals.Contains

::: actants.testing.evals.Predicate

::: actants.testing.evals.ToolCalled

::: actants.testing.evals.ToolsCalledInOrder

## Errors

::: actants.errors.RecordingError

::: actants.errors.RecordingFormatError

::: actants.errors.RecordingMissError

::: actants.errors.EvalError

::: actants.errors.ScorerError
