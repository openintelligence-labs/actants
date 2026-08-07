"""actants — local-first AI agent framework.

Public symbols are lazy-imported on first attribute access (PEP 562) so that
``import actants`` stays under 200ms regardless of which providers, embeddings,
storage, or interop modules a user actually touches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

# Public-attribute name → (module path, source attribute name).
_LAZY: dict[str, tuple[str, str]] = {
    # Agents
    "Agent": ("actants.agents.agent", "Agent"),
    "AgentResult": ("actants.agents.agent", "AgentResult"),
    "AgentStep": ("actants.agents.agent", "AgentStep"),
    "AgentHooks": ("actants.agents.hooks", "AgentHooks"),
    "ConversationMemory": ("actants.agents.memory", "ConversationMemory"),
    # Agent.stream() events. The LLM-level StreamEvent union and its members are
    # exported below; these are their agent-level counterparts and are what
    # Agent.stream() actually yields, so they belong at the same level.
    "AgentEvent": ("actants.agents.agent", "AgentEvent"),
    "AgentTextDelta": ("actants.agents.events", "AgentTextDelta"),
    "AgentToolCallStarted": ("actants.agents.events", "AgentToolCallStarted"),
    "AgentToolCallCompleted": ("actants.agents.events", "AgentToolCallCompleted"),
    "AgentStepCompleted": ("actants.agents.events", "AgentStepCompleted"),
    "AgentRunCompleted": ("actants.agents.events", "AgentRunCompleted"),
    # LLM core
    "LLM": ("actants.llm.client", "LLM"),
    "LLMSettings": ("actants.llm.client", "LLMSettings"),
    "BaseLLMProvider": ("actants.llm.base", "BaseLLMProvider"),
    "ChatMessage": ("actants.llm.base", "ChatMessage"),
    "CompletionResult": ("actants.llm.base", "CompletionResult"),
    "FinishDelta": ("actants.llm.base", "FinishDelta"),
    # The canonical vocabulary CompletionResult.finish_reason is drawn from. Exported
    # because it is the type of a public field: a consumer writing
    # `def handle(r: FinishReason) -> ...` needs to be able to name it.
    "FinishReason": ("actants.llm.finish_reason", "FinishReason"),
    "FINISH_REASONS": ("actants.llm.finish_reason", "FINISH_REASONS"),
    "normalize_finish_reason": ("actants.llm.finish_reason", "normalize_finish_reason"),
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
    "ToolResult": ("actants.tools.base", "ToolResult"),
    "ToolRegistry": ("actants.tools.registry", "ToolRegistry"),
    # Errors — users are expected to catch these, so they belong at the top level.
    "ActantsError": ("actants.errors", "ActantsError"),
    "ProviderError": ("actants.errors", "ProviderError"),
    "UnknownProviderError": ("actants.errors", "UnknownProviderError"),
    "ProviderNotInstalledError": ("actants.errors", "ProviderNotInstalledError"),
    "MissingAPIKeyError": ("actants.errors", "MissingAPIKeyError"),
    "ModelNotFoundError": ("actants.errors", "ModelNotFoundError"),
    "ToolCallsNotSupportedError": ("actants.errors", "ToolCallsNotSupportedError"),
    "ToolError": ("actants.tools.base", "ToolError"),
    "CacheSchemaMismatch": ("actants.cache.semantic", "CacheSchemaMismatch"),
    # Cache
    "InMemoryCache": ("actants.cache.memory", "InMemoryCache"),
    "CacheBackend": ("actants.cache.protocol", "CacheBackend"),
    "RequestCacheBackend": ("actants.cache.protocol", "RequestCacheBackend"),
    "CacheRequest": ("actants.cache.request", "CacheRequest"),
    # Cost
    "CostTracker": ("actants.cost.tracker", "CostTracker"),
    "PRICING": ("actants.cost.pricing", "PRICING"),
    "estimate_cost": ("actants.cost.pricing", "estimate_cost"),
    "estimate_cost_or_none": ("actants.cost.pricing", "estimate_cost_or_none"),
    "lookup_price": ("actants.cost.pricing", "lookup_price"),
    "is_priced": ("actants.cost.pricing", "is_priced"),
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
    # Type-checker only; re-exported at runtime via __getattr__ above.
    #
    # The redundant-looking `X as X` form is deliberate: it marks each name as an
    # explicit re-export (PEP 484). Without it, `__all__` being built dynamically
    # from `_LAZY` leaves type checkers unable to see these as public, and every
    # `from actants import LLM` fails under `mypy --strict` with
    # "does not explicitly export attribute".
    from actants.agents.agent import Agent as Agent
    from actants.agents.agent import AgentEvent as AgentEvent
    from actants.agents.agent import AgentResult as AgentResult
    from actants.agents.agent import AgentStep as AgentStep
    from actants.agents.events import AgentRunCompleted as AgentRunCompleted
    from actants.agents.events import AgentStepCompleted as AgentStepCompleted
    from actants.agents.events import AgentTextDelta as AgentTextDelta
    from actants.agents.events import AgentToolCallCompleted as AgentToolCallCompleted
    from actants.agents.events import AgentToolCallStarted as AgentToolCallStarted
    from actants.agents.hooks import AgentHooks as AgentHooks
    from actants.agents.memory import ConversationMemory as ConversationMemory
    from actants.cache.memory import InMemoryCache as InMemoryCache
    from actants.cache.protocol import CacheBackend as CacheBackend
    from actants.cache.protocol import RequestCacheBackend as RequestCacheBackend
    from actants.cache.request import CacheRequest as CacheRequest
    from actants.cache.semantic import CacheSchemaMismatch as CacheSchemaMismatch
    from actants.config.paths import app_cache_dir as app_cache_dir
    from actants.config.paths import app_config_dir as app_config_dir
    from actants.config.paths import app_data_dir as app_data_dir
    from actants.config.settings import AppSettings as AppSettings
    from actants.cost.pricing import PRICING as PRICING
    from actants.cost.pricing import estimate_cost as estimate_cost
    from actants.cost.pricing import estimate_cost_or_none as estimate_cost_or_none
    from actants.cost.pricing import is_priced as is_priced
    from actants.cost.pricing import lookup_price as lookup_price
    from actants.cost.tracker import CostTracker as CostTracker
    from actants.embeddings.base import BaseEmbeddingProvider as BaseEmbeddingProvider
    from actants.embeddings.base import EmbeddingResult as EmbeddingResult
    from actants.embeddings.client import Embeddings as Embeddings
    from actants.embeddings.client import EmbeddingSettings as EmbeddingSettings
    from actants.embeddings.ollama import OllamaEmbeddingProvider as OllamaEmbeddingProvider
    from actants.errors import ActantsError as ActantsError
    from actants.errors import MissingAPIKeyError as MissingAPIKeyError
    from actants.errors import ModelNotFoundError as ModelNotFoundError
    from actants.errors import ProviderError as ProviderError
    from actants.errors import ProviderNotInstalledError as ProviderNotInstalledError
    from actants.errors import ToolCallsNotSupportedError as ToolCallsNotSupportedError
    from actants.errors import UnknownProviderError as UnknownProviderError
    from actants.llm.base import BaseLLMProvider as BaseLLMProvider
    from actants.llm.base import ChatMessage as ChatMessage
    from actants.llm.base import CompletionResult as CompletionResult
    from actants.llm.base import FinishDelta as FinishDelta
    from actants.llm.base import StreamEvent as StreamEvent
    from actants.llm.base import TextDelta as TextDelta
    from actants.llm.base import TokenUsage as TokenUsage
    from actants.llm.base import ToolCall as ToolCall
    from actants.llm.base import ToolCallDelta as ToolCallDelta
    from actants.llm.base import ToolSpec as ToolSpec
    from actants.llm.base import UsageDelta as UsageDelta
    from actants.llm.client import LLM as LLM
    from actants.llm.client import LLMSettings as LLMSettings
    from actants.llm.finish_reason import FINISH_REASONS as FINISH_REASONS
    from actants.llm.finish_reason import FinishReason as FinishReason
    from actants.llm.finish_reason import normalize_finish_reason as normalize_finish_reason
    from actants.llm.ollama import OllamaProvider as OllamaProvider
    from actants.observability.logging import get_logger as get_logger
    from actants.observability.logging import setup_logging as setup_logging
    from actants.policies.fallback import AllProvidersFailedError as AllProvidersFailedError
    from actants.policies.fallback import FallbackProvider as FallbackProvider
    from actants.policies.retry import RetryPolicy as RetryPolicy
    from actants.policies.retry import retry_async as retry_async
    from actants.storage.jsonl import JsonlAppender as JsonlAppender
    from actants.storage.jsonl import read_jsonl as read_jsonl
    from actants.storage.sqlite import open_sqlite as open_sqlite
    from actants.tools.base import Tool as Tool
    from actants.tools.base import ToolError as ToolError
    from actants.tools.base import ToolResult as ToolResult
    from actants.tools.registry import ToolRegistry as ToolRegistry
    from actants.tracing.otel import get_tracer as get_tracer
    from actants.tracing.otel import instrument_llm as instrument_llm
    from actants.tracing.otel import llm_span as llm_span
