"""Use an :class:`~actants.agents.agent.Agent` as a graph node.

An Agent is a linear ReAct loop; a graph is the control flow *around* such loops. Making
one a node of the other is the point where the two halves of the framework meet, so it
should cost one line::

    graph.add_node("research", agent_node(researcher, prompt=lambda s: s.question,
                                          output="findings"))

``prompt`` reads the state to build the agent's user turn, and ``output`` names the state
field its answer lands in. If that field is annotated
:data:`~actants.graph.state.Append`, the answers accumulate across loop passes for free —
which is what makes an agent usable inside a retry or critique cycle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from actants.graph.state import StateT

if TYPE_CHECKING:
    from actants.agents.agent import Agent


def agent_node(
    agent: Agent,
    *,
    prompt: Callable[[StateT], str],
    output: str,
) -> Callable[[StateT], Awaitable[dict[str, Any]]]:
    """Adapt ``agent`` into a node function for a graph over ``StateT``.

    The returned node runs one agent turn per visit and writes the agent's final content
    to the ``output`` field.

    The agent's own ``thread_id`` is deliberately not wired to the graph's: the graph
    checkpoints the *state* after the node completes, which is the boundary that matters
    for "this node already ran". Giving the agent its own durable thread as well would
    make a resumed graph replay a half-finished agent turn inside a node the graph has
    already recorded as done. Pass a checkpointed Agent explicitly if you want the inner
    loop durable too, and give it a thread id derived from your own state.
    """

    async def run_agent(state: StateT) -> dict[str, Any]:
        result = await agent.run(prompt(state))
        return {output: result.content}

    return run_agent


__all__ = ["agent_node"]
