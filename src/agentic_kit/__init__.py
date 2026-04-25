from __future__ import annotations

from agentic_kit.cache.memory import InMemoryCache
from agentic_kit.cache.protocol import CacheBackend
from agentic_kit.cost.pricing import PRICING, estimate_cost
from agentic_kit.cost.tracker import CostTracker
from agentic_kit.llm.base import (
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
from agentic_kit.llm.client import LLM, LLMSettings
from agentic_kit.llm.ollama import OllamaProvider
from agentic_kit.policies.fallback import AllProvidersFailedError, FallbackProvider
from agentic_kit.policies.retry import RetryPolicy, retry_async
from agentic_kit.tools.base import Tool, ToolError, ToolResult
from agentic_kit.tools.registry import ToolRegistry
from agentic_kit.tracing.otel import get_tracer, instrument_llm, llm_span

__version__ = "0.3.0"

__all__ = [
    "AllProvidersFailedError",
    "BaseLLMProvider",
    "CacheBackend",
    "ChatMessage",
    "CompletionResult",
    "CostTracker",
    "FallbackProvider",
    "FinishDelta",
    "InMemoryCache",
    "LLM",
    "LLMSettings",
    "OllamaProvider",
    "PRICING",
    "RetryPolicy",
    "StreamEvent",
    "TextDelta",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "UsageDelta",
    "__version__",
    "estimate_cost",
    "get_tracer",
    "instrument_llm",
    "llm_span",
    "retry_async",
]
