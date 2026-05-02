"""agentic-kit — local-first AI agent framework.

Public symbols are lazy-imported on first attribute access (PEP 562) so that
``import agentic_kit`` stays under 200ms regardless of which providers, embeddings,
storage, or interop modules a user actually touches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.5.0"

# Map of public-attribute name → (module path, source attribute name).
# When user accesses ``agentic_kit.Agent`` for the first time, we import that
# module and bind the attribute on the package, so subsequent accesses are
# zero-overhead.
_LAZY: dict[str, tuple[str, str]] = {
    # Agents
    "Agent": ("agentic_kit.agents.agent", "Agent"),
    "AgentResult": ("agentic_kit.agents.agent", "AgentResult"),
    "AgentStep": ("agentic_kit.agents.agent", "AgentStep"),
    "AgentHooks": ("agentic_kit.agents.hooks", "AgentHooks"),
    "ConversationMemory": ("agentic_kit.agents.memory", "ConversationMemory"),
    # LLM core (kept eager-ish since most apps need it; lives behind lazy
    # access too so importing only ``agentic_kit.storage`` doesn't pay for it).
    "LLM": ("agentic_kit.llm.client", "LLM"),
    "LLMSettings": ("agentic_kit.llm.client", "LLMSettings"),
    "BaseLLMProvider": ("agentic_kit.llm.base", "BaseLLMProvider"),
    "ChatMessage": ("agentic_kit.llm.base", "ChatMessage"),
    "CompletionResult": ("agentic_kit.llm.base", "CompletionResult"),
    "FinishDelta": ("agentic_kit.llm.base", "FinishDelta"),
    "StreamEvent": ("agentic_kit.llm.base", "StreamEvent"),
    "TextDelta": ("agentic_kit.llm.base", "TextDelta"),
    "TokenUsage": ("agentic_kit.llm.base", "TokenUsage"),
    "ToolCall": ("agentic_kit.llm.base", "ToolCall"),
    "ToolCallDelta": ("agentic_kit.llm.base", "ToolCallDelta"),
    "ToolSpec": ("agentic_kit.llm.base", "ToolSpec"),
    "UsageDelta": ("agentic_kit.llm.base", "UsageDelta"),
    "OllamaProvider": ("agentic_kit.llm.ollama", "OllamaProvider"),
    # Tools
    "Tool": ("agentic_kit.tools.base", "Tool"),
    "ToolError": ("agentic_kit.tools.base", "ToolError"),
    "ToolResult": ("agentic_kit.tools.base", "ToolResult"),
    "ToolRegistry": ("agentic_kit.tools.registry", "ToolRegistry"),
    # Cache
    "InMemoryCache": ("agentic_kit.cache.memory", "InMemoryCache"),
    "CacheBackend": ("agentic_kit.cache.protocol", "CacheBackend"),
    # Cost
    "CostTracker": ("agentic_kit.cost.tracker", "CostTracker"),
    "PRICING": ("agentic_kit.cost.pricing", "PRICING"),
    "estimate_cost": ("agentic_kit.cost.pricing", "estimate_cost"),
    # Policies
    "RetryPolicy": ("agentic_kit.policies.retry", "RetryPolicy"),
    "retry_async": ("agentic_kit.policies.retry", "retry_async"),
    "FallbackProvider": ("agentic_kit.policies.fallback", "FallbackProvider"),
    "AllProvidersFailedError": ("agentic_kit.policies.fallback", "AllProvidersFailedError"),
    # Tracing
    "get_tracer": ("agentic_kit.tracing.otel", "get_tracer"),
    "instrument_llm": ("agentic_kit.tracing.otel", "instrument_llm"),
    "llm_span": ("agentic_kit.tracing.otel", "llm_span"),
    # Config
    "AppSettings": ("agentic_kit.config.settings", "AppSettings"),
    "app_cache_dir": ("agentic_kit.config.paths", "app_cache_dir"),
    "app_config_dir": ("agentic_kit.config.paths", "app_config_dir"),
    "app_data_dir": ("agentic_kit.config.paths", "app_data_dir"),
    # Observability
    "setup_logging": ("agentic_kit.observability.logging", "setup_logging"),
    "get_logger": ("agentic_kit.observability.logging", "get_logger"),
    # Embeddings
    "Embeddings": ("agentic_kit.embeddings.client", "Embeddings"),
    "EmbeddingResult": ("agentic_kit.embeddings.base", "EmbeddingResult"),
    "EmbeddingSettings": ("agentic_kit.embeddings.client", "EmbeddingSettings"),
    "BaseEmbeddingProvider": ("agentic_kit.embeddings.base", "BaseEmbeddingProvider"),
    "OllamaEmbeddingProvider": ("agentic_kit.embeddings.ollama", "OllamaEmbeddingProvider"),
    # Storage
    "open_sqlite": ("agentic_kit.storage.sqlite", "open_sqlite"),
    "JsonlAppender": ("agentic_kit.storage.jsonl", "JsonlAppender"),
    "read_jsonl": ("agentic_kit.storage.jsonl", "read_jsonl"),
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
    raise AttributeError(f"module 'agentic_kit' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    # Imports for static type-checkers only — never executed at runtime.
    # Re-exported via __getattr__ above.
    from agentic_kit.agents.agent import Agent, AgentResult, AgentStep  # noqa: F401
    from agentic_kit.agents.hooks import AgentHooks  # noqa: F401
    from agentic_kit.agents.memory import ConversationMemory  # noqa: F401
    from agentic_kit.cache.memory import InMemoryCache  # noqa: F401
    from agentic_kit.cache.protocol import CacheBackend  # noqa: F401
    from agentic_kit.config.paths import (  # noqa: F401
        app_cache_dir,
        app_config_dir,
        app_data_dir,
    )
    from agentic_kit.config.settings import AppSettings  # noqa: F401
    from agentic_kit.cost.pricing import PRICING, estimate_cost  # noqa: F401
    from agentic_kit.cost.tracker import CostTracker  # noqa: F401
    from agentic_kit.embeddings.base import (  # noqa: F401
        BaseEmbeddingProvider,
        EmbeddingResult,
    )
    from agentic_kit.embeddings.client import Embeddings, EmbeddingSettings  # noqa: F401
    from agentic_kit.embeddings.ollama import OllamaEmbeddingProvider  # noqa: F401
    from agentic_kit.llm.base import (  # noqa: F401
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
    from agentic_kit.llm.client import LLM, LLMSettings  # noqa: F401
    from agentic_kit.llm.ollama import OllamaProvider  # noqa: F401
    from agentic_kit.observability.logging import get_logger, setup_logging  # noqa: F401
    from agentic_kit.policies.fallback import (  # noqa: F401
        AllProvidersFailedError,
        FallbackProvider,
    )
    from agentic_kit.policies.retry import RetryPolicy, retry_async  # noqa: F401
    from agentic_kit.storage.jsonl import JsonlAppender, read_jsonl  # noqa: F401
    from agentic_kit.storage.sqlite import open_sqlite  # noqa: F401
    from agentic_kit.tools.base import Tool, ToolError, ToolResult  # noqa: F401
    from agentic_kit.tools.registry import ToolRegistry  # noqa: F401
    from agentic_kit.tracing.otel import get_tracer, instrument_llm, llm_span  # noqa: F401
