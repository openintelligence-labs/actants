from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    StreamEvent,
    ToolSpec,
)

log = structlog.get_logger(__name__)


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in a FallbackProvider chain fails."""

    def __init__(self, errors: list[tuple[str, BaseException]]) -> None:
        self.errors = errors
        parts = [f"{name}: {exc!r}" for name, exc in errors]
        super().__init__("All providers failed: " + "; ".join(parts))


class FallbackProvider(BaseLLMProvider):
    """Try each provider in order; on error, fall back to the next.

    Each entry is a (provider, model) pair so the caller can route an Ollama-first chain
    to an OpenAI backup that uses a different model name. If a model is ``None`` the
    caller-supplied model name is used for that provider.
    """

    name = "fallback"

    def __init__(
        self,
        chain: list[tuple[BaseLLMProvider, str | None]],
    ) -> None:
        if not chain:
            raise ValueError("FallbackProvider requires at least one provider")
        self._chain = chain
        # A chain is only as capable as its weakest link: any provider may end up
        # serving the request, so a capability holds only if all of them support it.
        self.supports_tool_calls = all(p.supports_tool_calls for p, _ in chain)
        self.supports_streaming_tools = all(p.supports_streaming_tools for p, _ in chain)

    async def health(self) -> bool:
        for provider, _ in self._chain:
            if await provider.health():
                return True
        return False

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        errors: list[tuple[str, BaseException]] = []
        for provider, pmodel in self._chain:
            try:
                return await provider.complete(
                    messages=messages,
                    model=pmodel or model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as exc:
                log.warning(
                    "fallback_provider_failed",
                    provider=provider.name,
                    error=str(exc),
                )
                errors.append((provider.name, exc))
        raise AllProvidersFailedError(errors)

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        errors: list[tuple[str, BaseException]] = []
        for provider, pmodel in self._chain:
            emitted = False
            try:
                agen = provider.stream(
                    messages=messages,
                    model=pmodel or model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                async for chunk in agen:
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted:
                    # Bytes already reached the consumer. Restarting on another provider
                    # would splice two different completions into one response, so the
                    # failure has to surface instead.
                    raise
                log.warning(
                    "fallback_stream_failed",
                    provider=provider.name,
                    error=str(exc),
                )
                errors.append((provider.name, exc))
        raise AllProvidersFailedError(errors)

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Fall back across providers while preserving typed events.

        Without this override the base-class implementation would wrap each provider's
        ``stream`` (text only), silently dropping tool-call and usage events for every
        provider in the chain.
        """
        errors: list[tuple[str, BaseException]] = []
        for provider, pmodel in self._chain:
            emitted = False
            try:
                agen = provider.stream_events(
                    messages=messages,
                    model=pmodel or model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs,
                )
                async for event in agen:
                    emitted = True
                    yield event
                return
            except Exception as exc:
                if emitted:
                    raise
                log.warning(
                    "fallback_stream_failed",
                    provider=provider.name,
                    error=str(exc),
                )
                errors.append((provider.name, exc))
        raise AllProvidersFailedError(errors)
