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

This module has no imports from the rest of actants, so any subpackage can raise from
it without risking an import cycle.

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
    └── MCPConnectionError              (RuntimeError)
"""

from __future__ import annotations

__all__ = [
    "ActantsError",
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderNotInstalledError",
    "ToolCallsNotSupportedError",
    "UnknownProviderError",
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
