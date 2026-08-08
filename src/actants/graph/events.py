"""Streaming event taxonomy for ``CompiledGraph.stream()``.

The graph-level counterpart of `events`: same dataclass shape, same
``match``-friendly ergonomics, named for graph vocabulary (nodes and iterations) rather
than agent vocabulary (steps and tool calls).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class GraphNodeStarted:
    """A node is about to run. ``iteration`` counts nodes executed in this run."""

    node: str
    iteration: int


@dataclass
class GraphNodeCompleted[StateT: BaseModel]:
    """A node returned and its update was merged. ``state`` is the result of that merge."""

    node: str
    iteration: int
    state: StateT


@dataclass
class GraphInterrupted[StateT: BaseModel]:
    """The run paused in front of an ``interrupt_before`` node and persisted.

    ``node`` is the node that has *not* run. Resume with
    ``compiled.resume(thread_id, approve=True)`` to run it, or ``approve=False`` to skip
    it and continue past it.
    """

    node: str
    state: StateT


@dataclass
class GraphCompleted[StateT: BaseModel]:
    """The run reached END. ``state`` is the final state (terminal event)."""

    state: StateT


#: Everything ``CompiledGraph.stream()`` yields.
type GraphEvent[StateT: BaseModel] = (
    GraphNodeStarted
    | GraphNodeCompleted[StateT]
    | GraphInterrupted[StateT]
    | GraphCompleted[StateT]
)


__all__ = [
    "GraphCompleted",
    "GraphEvent",
    "GraphInterrupted",
    "GraphNodeCompleted",
    "GraphNodeStarted",
]
