"""Actionable exceptions for common provider failures.

Every error raised here names the exact problem and the exact fix. The rule is:
if a user can hit it in their first ten minutes, the message must tell them what
to type next.
"""

from __future__ import annotations

import contextlib

import httpx

__all__ = [
    "ActantsError",
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderNotInstalledError",
    "UnknownProviderError",
    "raise_for_ollama_error",
]


class ActantsError(Exception):
    """Base class for actants errors that carry a suggested fix."""


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


def _ollama_not_running(base_url: str, exc: Exception) -> ProviderError:
    return ProviderError(
        f"Cannot reach the Ollama server at {base_url}. "
        "Is Ollama running? Start it with `ollama serve`, or install it from "
        "https://ollama.com. "
        "To use a hosted provider instead, pass e.g. "
        "LLM(provider='openai', model='gpt-4o') with OPENAI_API_KEY set. "
        f"(underlying error: {type(exc).__name__}: {exc})"
    )


async def _installed_ollama_models(client: httpx.AsyncClient, base_url: str) -> list[str]:
    """Best-effort listing of models pulled on the Ollama server."""
    try:
        r = await client.get(f"{base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []) if m.get("name"))
    except Exception:
        return []


def _model_not_found(model: str, base_url: str, installed: list[str]) -> ModelNotFoundError:
    if installed:
        available = ", ".join(installed)
        hint = f"Installed models: {available}."
    else:
        hint = "No models are installed on that server."
    return ModelNotFoundError(
        f"Model {model!r} is not available on the Ollama server at {base_url}. "
        f"{hint} "
        f"Run `ollama pull {model}` to download it, or pass a model you already "
        "have, e.g. LLM(model='<one of the above>')."
    )


async def raise_for_ollama_error(
    exc: Exception,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
) -> None:
    """Translate a raw httpx failure from Ollama into an actionable actants error.

    Re-raises a :class:`ProviderError` subclass when the cause is recognisable
    (server down, model not pulled), otherwise returns so the caller can re-raise
    the original exception untouched.
    """
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        raise _ollama_not_running(base_url, exc) from exc

    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        # Ollama returns 404 from /api/chat both when the model is missing and
        # when the endpoint is wrong; the body distinguishes them.
        body = ""
        with contextlib.suppress(Exception):
            body = exc.response.text
        if "not found" in body.lower() or "model" in body.lower():
            installed = await _installed_ollama_models(client, base_url)
            raise _model_not_found(model, base_url, installed) from exc
