"""LLM gateway: the client, the shared message/completion model, and the providers.

Hosted providers are lazy-imported (PEP 562) rather than imported at module scope.
Importing them eagerly pulls their optional SDK in at ``import actants.llm`` time and
creates a cycle back through ``actants.cache.memory``, so they are resolved on first
attribute access instead. The ``TYPE_CHECKING`` block below re-exports them with the
``X as X`` form so ``from actants.llm import OpenAIProvider`` type-checks under
``mypy --strict`` — without it, only ``OllamaProvider`` was importable, which was an
arbitrary difference between the local provider and every hosted one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    Role,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)
from actants.llm.client import KNOWN_PROVIDERS, LLM, LLMSettings
from actants.llm.ollama import OllamaProvider

#: Public name → module providing it. Each needs an optional extra installed.
_LAZY_PROVIDERS: dict[str, str] = {
    "OpenAIProvider": "actants.llm.openai_provider",
    "AnthropicProvider": "actants.llm.anthropic_provider",
    "GeminiProvider": "actants.llm.gemini_provider",
    "GroqProvider": "actants.llm.groq_provider",
    "MistralProvider": "actants.llm.mistral_provider",
}

__all__ = [
    "KNOWN_PROVIDERS",
    "LLM",
    "AnthropicProvider",
    "BaseLLMProvider",
    "ChatMessage",
    "CompletionResult",
    "FinishDelta",
    "GeminiProvider",
    "GroqProvider",
    "LLMSettings",
    "MistralProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Role",
    "StreamEvent",
    "TextDelta",
    "TokenUsage",
    "ToolCall",
    "ToolCallDelta",
    "ToolSpec",
    "UsageDelta",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_PROVIDERS.get(name)
    if module_path is not None:
        from importlib import import_module

        value = getattr(import_module(module_path), name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module 'actants.llm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    # Type-checker only; resolved at runtime via __getattr__ above. The `X as X` form
    # marks each as an explicit re-export (PEP 484) so strict consumers can import them.
    from actants.llm.anthropic_provider import AnthropicProvider as AnthropicProvider
    from actants.llm.gemini_provider import GeminiProvider as GeminiProvider
    from actants.llm.groq_provider import GroqProvider as GroqProvider
    from actants.llm.mistral_provider import MistralProvider as MistralProvider
    from actants.llm.openai_provider import OpenAIProvider as OpenAIProvider
