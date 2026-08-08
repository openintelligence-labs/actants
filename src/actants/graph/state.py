"""Typed graph state: the END sentinel, reducers, and how an update is merged.

A graph's state is a user-defined pydantic model. A node returns a *partial* update —
a mapping of field name to new value — and this module decides how that update lands on
the state the next node will see.

The default is replace-per-field, which is what a plain assignment would do. The one
alternative worth building in is accumulation, because a graph that loops almost always
has a field it appends to (a message list, a scratchpad of findings). It is spelled with
:data:`Append` inside a :class:`typing.Annotated`::

    class State(BaseModel):
        question: str
        findings: Annotated[list[str], Append] = Field(default_factory=list)

A node returning ``{"findings": ["one"]}`` extends the list rather than replacing it, so
each pass through a loop adds to what the earlier passes found.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Literal, TypeVar

from pydantic import BaseModel

from actants.errors import GraphValidationError

#: The terminal node name. A router returning this — or an edge pointing at it — ends
#: the run and hands the current state back to the caller.
#:
#: Spelled as a ``Literal`` string rather than an object sentinel so it survives a JSON
#: round-trip into a checkpoint unchanged, and so a router's return type can be written
#: ``str`` without a union. The double underscores make a collision with a real node
#: name implausible, and :func:`~actants.graph.state_graph.StateGraph.add_node` rejects
#: it outright.
END: Final = "__end__"

#: Type of the END sentinel, for a router annotated ``-> str | Literal[END]``.
EndT = Literal["__end__"]

StateT = TypeVar("StateT", bound=BaseModel)


class _AppendReducer:
    """Marker for a field whose updates accumulate instead of replacing.

    A class rather than a bare string so ``Annotated[list[str], Append]`` is unambiguous:
    a string in the metadata could plausibly be someone else's annotation, and this must
    never be confused with one.
    """

    def __repr__(self) -> str:
        return "Append"


#: Reducer marker for accumulating fields; see the module docstring for the idiom.
Append: Final = _AppendReducer()


def append_fields(state_type: type[BaseModel]) -> frozenset[str]:
    """Return the names of ``state_type``'s fields marked :data:`Append`.

    Read off pydantic's own field metadata, so the annotation is the single source of
    truth — there is no parallel registry of reducers to fall out of sync with the model.
    """
    return frozenset(
        name
        for name, field in state_type.model_fields.items()
        if any(isinstance(m, _AppendReducer) for m in field.metadata)
    )


def validate_append_fields(state_type: type[BaseModel]) -> None:
    """Reject :data:`Append` on a field that cannot accumulate.

    Checked at compile time rather than on the first update, because a scalar marked
    ``Append`` is a mistake in the model definition and there is no reason to let a
    graph run for several nodes before saying so.
    """
    for name in sorted(append_fields(state_type)):
        annotation = state_type.model_fields[name].annotation
        origin = getattr(annotation, "__origin__", annotation)
        if not (isinstance(origin, type) and issubclass(origin, list)):
            raise GraphValidationError(
                f"Field {name!r} on {state_type.__name__} is annotated with Append, but "
                f"its type is {annotation!r}, which is not a list. The Append reducer "
                "extends a list with each update; a field that replaces needs no "
                "annotation at all. Write "
                f"`{name}: Annotated[list[...], Append] = Field(default_factory=list)`."
            )


def merge_update(
    state: StateT,
    update: dict[str, Any],
    *,
    accumulate: frozenset[str],
    node: str,
) -> StateT:
    """Apply one node's update to ``state``, returning a new state.

    Never mutates ``state``, and never *shares* mutable containers with it: the returned
    state is independent, so a node that mutates a list in place cannot reach back into
    the value an earlier node — or a concurrent run — is holding. ``model_copy`` alone is
    shallow and would leave the two aliased.

    Unknown keys are an error rather than being ignored. A node returning a dict cannot
    have its keys checked by a type checker, so this is the only place a typo like
    ``{"anwser": ...}`` can be caught — and silently dropping it would show up much
    later as a field that mysteriously never changed.
    """
    if not update:
        return state

    known = state.__class__.model_fields
    unknown = sorted(set(update) - set(known))
    if unknown:
        raise GraphValidationError(
            f"Node {node!r} returned key(s) {unknown} that are not fields of "
            f"{state.__class__.__name__}. Its fields are: {sorted(known)}. "
            "A node returns a partial update whose keys must be state field names."
        )

    merged: dict[str, Any] = {}
    for key, value in update.items():
        if key in accumulate:
            existing = getattr(state, key)
            merged[key] = [*existing, *_as_items(value, node=node, field=key)]
        else:
            merged[key] = value

    # model_copy(update=...) deliberately skips validation, which would otherwise re-run
    # every validator on every node and turn a field's validator into a per-node cost.
    # The values come from typed node functions, and unknown keys are rejected above.
    # deep=True because the shallow form aliases every container the update did not
    # replace, which is how an in-place mutation in one run reached another run's state.
    return state.model_copy(update=deepcopy(merged), deep=True)


def _as_items(value: Any, *, node: str, field: str) -> list[Any]:
    """Read a node's update to an Append field as the items to add.

    A list extends; anything else is taken as a single item. That asymmetry is
    deliberate — returning ``{"findings": "one"}`` for a ``list[str]`` field is common
    enough that treating it as one item beats raising, and a string is exactly the value
    that would otherwise be spread into characters.
    """
    if isinstance(value, list):
        return list(value)
    return [value]


def state_to_json(state: BaseModel) -> str:
    """Serialize graph state for a checkpoint."""
    return state.model_dump_json()


def state_from_json(state_type: type[StateT], payload: str) -> StateT:
    """Rebuild graph state from a checkpoint, validating it against the model.

    Validated rather than constructed, because the payload may have been written by a
    different process — and, on a schema change, by a different version of the model.
    """
    return state_type.model_validate_json(payload)


__all__ = [
    "END",
    "Append",
    "EndT",
    "StateT",
    "append_fields",
    "merge_update",
    "state_from_json",
    "state_to_json",
    "validate_append_fields",
]
