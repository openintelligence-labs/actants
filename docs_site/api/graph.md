# StateGraph

Branching and looping workflows over a typed state model. See
[StateGraph](../concepts/graph.md) for the walkthrough.

::: actants.graph.state_graph.StateGraph

::: actants.graph.state_graph.CompiledGraph

::: actants.graph.state_graph.GraphResult

## State

`END` is the terminal sentinel: return it from a router, or name it as a target in a
conditional mapping, to finish the run. Its type is exported as `EndT` for signatures
that mention it.

::: actants.graph.state.Append

## Events

::: actants.graph.events.GraphNodeStarted

::: actants.graph.events.GraphNodeCompleted

::: actants.graph.events.GraphInterrupted

::: actants.graph.events.GraphCompleted

## Agents as nodes

::: actants.graph.agent_node.agent_node

## Errors

::: actants.errors.GraphError

::: actants.errors.GraphValidationError

::: actants.errors.GraphRecursionError
