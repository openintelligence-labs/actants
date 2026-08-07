# Durability

Checkpoint storage for agent runs that must survive the process that started them.
See [Durability](../concepts/durability.md) for the replay guarantee and when it
applies.

::: actants.agents.checkpoint.Checkpointer

::: actants.agents.checkpoint.Checkpoint

::: actants.agents.checkpoint.StepRecord

## Implementations

::: actants.agents.checkpoint.InMemoryCheckpointer

::: actants.agents.checkpoint.SqliteCheckpointer

## Errors

::: actants.errors.CheckpointError

::: actants.errors.UnknownThreadError

::: actants.errors.UnresolvedToolCallError

::: actants.errors.CheckpointSchemaMismatch
