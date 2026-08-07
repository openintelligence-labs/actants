"""The exception hierarchy, in one place.

Every exception actants raises on purpose inherits from :class:`ActantsError`, so
``except ActantsError`` is a complete catch for "actants itself refused" — as opposed
to a bug in the caller's code or an error escaping from a provider SDK.

Each class also inherits the builtin exception a caller would naturally reach for, so
existing handlers keep working: :class:`UnknownProviderError` is a ``ValueError``,
:class:`ProviderNotInstalledError` is an ``ImportError``,
:class:`ToolCallsNotSupportedError` is a ``TypeError``. Catching either the actants
class or the builtin one works, and the narrower class is always available when a
caller wants to be precise::

    from actants import ActantsError, ModelNotFoundError

    try:
        result = await llm.complete("hi")
    except ModelNotFoundError as exc:
        print(f"pull the model first: {exc}")
    except ActantsError as exc:
        print(f"actants refused: {exc}")

This module has no *runtime* imports from the rest of actants, so any subpackage can
raise from it without risking an import cycle.

The hierarchy::

    ActantsError
    ├── ProviderError
    │   ├── UnknownProviderError        (ValueError)
    │   ├── ProviderNotInstalledError   (ImportError)
    │   ├── MissingAPIKeyError          (ValueError)
    │   ├── ModelNotFoundError          (ValueError)
    │   ├── ToolCallsNotSupportedError  (TypeError)
    │   └── AllProvidersFailedError     (RuntimeError)
    ├── ToolError
    ├── CacheSchemaMismatch             (RuntimeError)
    ├── CheckpointError
    │   ├── UnknownThreadError          (KeyError)
    │   ├── UnresolvedToolCallError     (RuntimeError)
    │   └── CheckpointSchemaMismatch    (RuntimeError)
    ├── GraphError
    │   ├── GraphValidationError        (ValueError)
    │   └── GraphRecursionError         (RuntimeError)
    └── MCPConnectionError              (RuntimeError)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actants.llm.base import ToolCall

__all__ = [
    "ActantsError",
    "CheckpointError",
    "CheckpointSchemaMismatch",
    "GraphError",
    "GraphRecursionError",
    "GraphValidationError",
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderNotInstalledError",
    "ToolCallsNotSupportedError",
    "UnknownProviderError",
    "UnsupportedSchemaError",
    "UnknownThreadError",
    "UnresolvedToolCallError",
]


class ActantsError(Exception):
    """Base class for every error actants raises deliberately.

    Catching this catches exactly the failures actants decided to report — a missing
    API key, an unknown provider, a model that is not pulled — and nothing else.
    """


class ProviderError(ActantsError):
    """A provider could not be reached or returned an unusable response."""


class UnknownProviderError(ProviderError, ValueError):
    """The requested provider name is not one actants knows about."""


class ProviderNotInstalledError(ProviderError, ImportError):
    """The provider's optional extra is not installed."""


class MissingAPIKeyError(ProviderError, ValueError):
    """The provider needs an API key and none was found."""


class ModelNotFoundError(ProviderError, ValueError):
    """The server is reachable but does not have the requested model."""


class ToolCallsNotSupportedError(ProviderError, TypeError):
    """Tools were passed to a provider that declares it cannot call them."""


class UnsupportedSchemaError(ProviderError, ValueError):
    """A schema cannot be expressed in a provider's native structured-output dialect.

    Raised inside the schema translators and handled by
    :meth:`~actants.llm.client.LLM.extract`, which downgrades that call to the
    prompt-based path rather than failing — so it does not normally reach a caller.
    Catch it when calling :func:`~actants.llm.structured.to_strict_schema` or
    :func:`~actants.llm.structured.to_gemini_schema` directly.
    """


class CheckpointError(ActantsError):
    """A durable agent run could not be persisted, found, or resumed."""


class UnknownThreadError(CheckpointError, KeyError):
    """No checkpoint exists for the requested ``thread_id``.

    Either the thread never ran under a checkpointer, or its state was deleted. Also a
    ``KeyError``, since the checkpointer is a keyed store.
    """

    def __str__(self) -> str:
        # KeyError.__str__ reprs its argument, which would turn a carefully worded
        # message into a quoted blob with escaped newlines.
        return str(self.args[0]) if self.args else ""


class UnresolvedToolCallError(CheckpointError, RuntimeError):
    """Resume hit an in-flight call to a tool that declared ``idempotent=False``.

    The process died while this call was executing, so actants cannot know whether the
    side effect happened. Re-dispatching could duplicate it; skipping could drop it.
    Rather than guess, resume raises this and hands the decision back — see
    :meth:`~actants.agents.agent.Agent.resume` for the ``resolve=`` options.

    The unresolved call is on :attr:`call`, and the thread it belongs to on
    :attr:`thread_id`, so a handler can present it to a human or look it up in a
    vendor's API by :attr:`ToolCall.id <actants.llm.base.ToolCall.id>`.
    """

    def __init__(self, message: str, *, thread_id: str, call: ToolCall) -> None:
        super().__init__(message)
        self.thread_id = thread_id
        self.call = call


class CheckpointSchemaMismatch(CheckpointError, RuntimeError):
    """An on-disk checkpoint store was written by an incompatible schema version.

    Unlike a cache, this is never discarded automatically: the rows are the only record
    of which tool side effects already ran.
    """


class GraphError(ActantsError):
    """A :class:`~actants.graph.state_graph.StateGraph` could not be built or run."""


class GraphValidationError(GraphError, ValueError):
    """A graph's shape is wrong, caught by ``compile()`` before anything runs.

    Every check this reports — an undefined edge target, an unreachable node, a missing
    entry point — is a structural mistake that would otherwise surface as a confusing
    failure halfway through a run, after side effects had already happened.
    """


class GraphRecursionError(GraphError, RuntimeError):
    """A graph ran ``max_iterations`` nodes without reaching END.

    Graphs loop by design, so actants cannot tell a slow convergence from a stuck one;
    the cap is the backstop. The message names the node the run was executing when the
    budget ran out, which is nearly always the one whose router never returns END.
    """

    def __init__(self, message: str, *, node: str, iterations: int) -> None:
        super().__init__(message)
        self.node = node
        self.iterations = iterations
