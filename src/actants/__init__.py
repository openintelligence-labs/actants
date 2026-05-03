"""actants — local-first AI agent framework.

Public symbols are lazy-imported on first attribute access (PEP 562) so that
``import actants`` stays under 200ms regardless of which providers, embeddings,
storage, or interop modules a user actually touches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.5.0"

# Map of public-attribute name → (module path, source attribute name).
# When user accesses ``actants.Agent`` for the first time, we import that
# module and bind the attribute on the package, so subsequent accesses are
# zero-overhead.
_LAZY: dict[str, tuple[str, str]] = {
    # Agents
    "Agent": ("actants.agents.agent", "Agent"),
    "AgentResult": ("actants.agents.agent", "AgentResult"),
    "AgentStep": ("actants.agents.agent", "AgentStep"),
    "AgentHooks": ("actants.agents.hooks", "AgentHooks"),
    "ConversationMemory": ("actants.agents.memory", "ConversationMemory"),
    # LLM core (kept eager-ish since most apps need it; lives behind lazy
    # access too so importing only ``actants.storage`` doesn't pay for it).
    "LLM": ("actants.llm.client", "LLM"),
    "LLMSettings": ("actants.llm.client", "LLMSettings"),
    "BaseLLMProvider": ("actants.llm.base", "BaseLLMProvider"),
    "ChatMessage": ("actants.llm.base", "ChatMessage"),
    "CompletionResult": ("actants.llm.base", "CompletionResult"),
    "FinishDelta": ("actants.llm.base", "FinishDelta"),
    "StreamEvent": ("actants.llm.base", "StreamEvent"),
    "TextDelta": ("actants.llm.base", "TextDelta"),
    "TokenUsage": ("actants.llm.base", "TokenUsage"),
    "ToolCall": ("actants.llm.base", "ToolCall"),
    "ToolCallDelta": ("actants.llm.base", "ToolCallDelta"),
    "ToolSpec": ("actants.llm.base", "ToolSpec"),
    "UsageDelta": ("actants.llm.base", "UsageDelta"),
    "OllamaProvider": ("actants.llm.ollama", "OllamaProvider"),
    # Tools
    "Tool": ("actants.tools.base", "Tool"),
    "ToolError": ("actants.tools.base", "ToolError"),
    "ToolResult": ("actants.tools.base", "ToolResult"),
    "ToolRegistry": ("actants.tools.registry", "ToolRegistry"),
    # Cache
    "InMemoryCache": ("actants.cache.memory", "InMemoryCache"),
    "CacheBackend": ("actants.cache.protocol", "CacheBackend"),
    # Cost
    "CostTracker": ("actants.cost.tracker", "CostTracker"),
    "PRICING": ("actants.cost.pricing", "PRICING"),
    "estimate_cost": ("actants.cost.pricing", "estimate_cost"),
    # Policies
    "RetryPolicy": ("actants.policies.retry", "RetryPolicy"),
    "retry_async": ("actants.policies.retry", "retry_async"),
    "FallbackProvider": ("actants.policies.fallback", "FallbackProvider"),
    "AllProvidersFailedError": ("actants.policies.fallback", "AllProvidersFailedError"),
    # Tracing
    "get_tracer": ("actants.tracing.otel", "get_tracer"),
    "instrument_llm": ("actants.tracing.otel", "instrument_llm"),
    "llm_span": ("actants.tracing.otel", "llm_span"),
    # Config
    "AppSettings": ("actants.config.settings", "AppSettings"),
    "app_cache_dir": ("actants.config.paths", "app_cache_dir"),
    "app_config_dir": ("actants.config.paths", "app_config_dir"),
    "app_data_dir": ("actants.config.paths", "app_data_dir"),
    # Observability
    "setup_logging": ("actants.observability.logging", "setup_logging"),
    "get_logger": ("actants.observability.logging", "get_logger"),
    # Embeddings
    "Embeddings": ("actants.embeddings.client", "Embeddings"),
    "EmbeddingResult": ("actants.embeddings.base", "EmbeddingResult"),
    "EmbeddingSettings": ("actants.embeddings.client", "EmbeddingSettings"),
    "BaseEmbeddingProvider": ("actants.embeddings.base", "BaseEmbeddingProvider"),
    "OllamaEmbeddingProvider": ("actants.embeddings.ollama", "OllamaEmbeddingProvider"),
    # Storage
    "open_sqlite": ("actants.storage.sqlite", "open_sqlite"),
    "JsonlAppender": ("actants.storage.jsonl", "JsonlAppender"),
    "read_jsonl": ("actants.storage.jsonl", "read_jsonl"),
}

__all__ = sorted([*_LAZY.keys(), "__version__"])


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        from importlib import import_module

        module = import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module 'actants' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    # Imports for static type-checkers only — never executed at runtime.
    # Re-exported via __getattr__ above.
    from actants.agents.agent import Agent, AgentResult, AgentStep  # noqa: F401
    from actants.agents.hooks import AgentHooks  # noqa: F401
    from actants.agents.memory import ConversationMemory  # noqa: F401
    from actants.cache.memory import InMemoryCache  # noqa: F401
    from actants.cache.protocol import CacheBackend  # noqa: F401
    from actants.config.paths import (  # noqa: F401
        app_cache_dir,
        app_config_dir,
        app_data_dir,
    )
    from actants.config.settings import AppSettings  # noqa: F401
    from actants.cost.pricing import PRICING, estimate_cost  # noqa: F401
    from actants.cost.tracker import CostTracker  # noqa: F401
    from actants.embeddings.base import (  # noqa: F401
        BaseEmbeddingProvider,
        EmbeddingResult,
    )
    from actants.embeddings.client import Embeddings, EmbeddingSettings  # noqa: F401
    from actants.embeddings.ollama import OllamaEmbeddingProvider  # noqa: F401
    from actants.llm.base import (  # noqa: F401
        BaseLLMProvider,
        ChatMessage,
        CompletionResult,
        FinishDelta,
        StreamEvent,
        TextDelta,
        TokenUsage,
        ToolCall,
        ToolCallDelta,
        ToolSpec,
        UsageDelta,
    )
    from actants.llm.client import LLM, LLMSettings  # noqa: F401
    from actants.llm.ollama import OllamaProvider  # noqa: F401
    from actants.observability.logging import get_logger, setup_logging  # noqa: F401
    from actants.policies.fallback import (  # noqa: F401
        AllProvidersFailedError,
        FallbackProvider,
    )
    from actants.policies.retry import RetryPolicy, retry_async  # noqa: F401
    from actants.storage.jsonl import JsonlAppender, read_jsonl  # noqa: F401
    from actants.storage.sqlite import open_sqlite  # noqa: F401
    from actants.tools.base import Tool, ToolError, ToolResult  # noqa: F401
    from actants.tools.registry import ToolRegistry  # noqa: F401
    from actants.tracing.otel import get_tracer, instrument_llm, llm_span  # noqa: F401
