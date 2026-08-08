from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from actants.errors import ProviderError
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    StreamEvent,
    ToolSpec,
)

log = structlog.get_logger(__name__)


class AllProvidersFailedError(ProviderError, RuntimeError):
    """Raised when every provider in a FallbackProvider chain fails.

    Carries the per-provider failures in ``errors`` so a caller can inspect why each
    link gave up, not just that the chain did.
    """

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
            raise ValueError(
                "FallbackProvider requires at least one provider, got an empty chain. "
                "Pass a list of (provider, model) pairs: "
                "FallbackProvider([(OllamaProvider(), 'llama3.2'), (openai, 'gpt-4o')])."
            )
        self._chain = chain

    @property
    def supports_tool_calls(self) -> bool:
        """True only if *every* provider in the chain supports tool calls.

        A chain is only as capable as its weakest link: any provider may end up serving
        the request, so a capability holds only if all of them support it.

        Derived on access rather than snapshotted in ``__init__`` because a provider's
        flag is an ordinary attribute that callers legitimately set after construction —
        a provider that learns its capabilities from a handshake, or a test that flips
        the flag. A construction-time snapshot silently kept the old answer, so tools
        were either refused by a chain that could serve them or passed to one that
        could not.
        """
        return all(p.supports_tool_calls for p, _ in self._chain)

    @supports_tool_calls.setter
    def supports_tool_calls(self, value: bool) -> None:
        raise AttributeError(
            "FallbackProvider.supports_tool_calls is derived from the chain and cannot "
            "be assigned — it is True only when every provider in the chain supports "
            "tool calls. Set the flag on the provider that is wrong instead, e.g. "
            "`my_provider.supports_tool_calls = True`."
        )

    @property
    def supports_streaming_tools(self) -> bool:
        """True only if *every* provider in the chain supports streaming tool calls.

        Derived on access, for the same reason as `supports_tool_calls`.
        """
        return all(p.supports_streaming_tools for p, _ in self._chain)

    @supports_streaming_tools.setter
    def supports_streaming_tools(self, value: bool) -> None:
        raise AttributeError(
            "FallbackProvider.supports_streaming_tools is derived from the chain and "
            "cannot be assigned — it is True only when every provider in the chain "
            "supports streaming tool calls. Set the flag on the provider that is wrong "
            "instead, e.g. `my_provider.supports_streaming_tools = True`."
        )

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

        This is the only streaming method the chain implements: ``stream`` is derived
        from it by `BaseLLMProvider`, so plain-text streaming
        gets the same fallback behaviour for free.

        Fallback stops as soon as the first event reaches the consumer. Restarting on
        another provider after that would splice two different completions into one
        response, so a mid-stream failure surfaces instead of being retried.
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
