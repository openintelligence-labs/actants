"""The OpenAI-compatible provider family.

xAI, DeepSeek, Together, Fireworks, OpenRouter, Cerebras, and Perplexity are the same
provider pointed at different hosts. These tests pin the two things that can silently
break: the base URL each one talks to, and the fact that every one of them is
constructible, registered, and reports actionable errors.

No test here makes a network call — the OpenAI SDK is driven against a stubbed
transport, so the suite runs with no API keys present.
"""

from __future__ import annotations

import json

import httpx
import pytest

from actants import LLM, LLMSettings
from actants.llm.client import _PROVIDER_REQUIREMENTS, KNOWN_PROVIDERS
from actants.llm.errors import MissingAPIKeyError, UnknownProviderError
from actants.llm.openai_compatible import (
    OPENAI_COMPATIBLE_PROVIDERS,
    CerebrasProvider,
    DeepSeekProvider,
    FireworksProvider,
    OpenRouterProvider,
    PerplexityProvider,
    TogetherProvider,
    XAIProvider,
)

pytest.importorskip("openai")

NEW_PROVIDERS = (
    "xai",
    "deepseek",
    "together",
    "fireworks",
    "openrouter",
    "cerebras",
    "perplexity",
)

ALL_ENV_VARS = [v for v, _ in _PROVIDER_REQUIREMENTS.values() if v]


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch):
    """Never let a developer's real key leak into these tests."""
    for var in [*ALL_ENV_VARS, "ACTANTS_API_KEY"]:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_provider_is_registered(provider: str):
    assert provider in KNOWN_PROVIDERS
    assert provider in _PROVIDER_REQUIREMENTS


def test_every_registered_provider_can_actually_be_built():
    """The guard against a name in the requirements table with no constructor.

    `_make_provider` used to end in a bare `return MistralProvider(...)`, so any
    provider added to the table but nowhere else would have been silently constructed
    as Mistral — a wrong-endpoint bug with no error.
    """
    native = {"ollama", "openai", "anthropic", "gemini"}
    for provider in KNOWN_PROVIDERS:
        if provider in native:
            continue
        assert provider in OPENAI_COMPATIBLE_PROVIDERS, (
            f"{provider!r} is registered in _PROVIDER_REQUIREMENTS but has no entry in "
            "OPENAI_COMPATIBLE_PROVIDERS, so _make_provider cannot build it."
        )


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_llm_constructs_the_right_provider(provider: str):
    llm = LLM(settings=LLMSettings(provider=provider, api_key="test-key", model="m"))
    assert llm.provider.name == provider


# --------------------------------------------------------------------------
# Base URLs — the one thing that distinguishes these providers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (XAIProvider, "https://api.x.ai/v1"),
        (DeepSeekProvider, "https://api.deepseek.com/v1"),
        (TogetherProvider, "https://api.together.xyz/v1"),
        (FireworksProvider, "https://api.fireworks.ai/inference/v1"),
        (OpenRouterProvider, "https://openrouter.ai/api/v1"),
        (CerebrasProvider, "https://api.cerebras.ai/v1"),
        (PerplexityProvider, "https://api.perplexity.ai"),
    ],
)
def test_default_base_url(cls: type, expected: str):
    provider = cls(api_key="test-key")
    assert str(provider._client.base_url).rstrip("/") == expected.rstrip("/")


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_base_url_is_overridable(provider: str):
    """Self-hosted gateways and proxies are the common reason to point one elsewhere."""
    from actants.llm.openai_compatible import openai_compatible_provider

    cls = openai_compatible_provider(provider, *OPENAI_COMPATIBLE_PROVIDERS[provider])
    built = cls(api_key="k", base_url="http://localhost:9999/v1")
    assert str(built._client.base_url).rstrip("/") == "http://localhost:9999/v1"


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_accepts_an_injected_client(provider: str):
    """Narrowing the parent signature made `Provider(client=...)` a TypeError once."""
    from openai import AsyncOpenAI

    from actants.llm.openai_compatible import openai_compatible_provider

    cls = openai_compatible_provider(provider, *OPENAI_COMPATIBLE_PROVIDERS[provider])
    injected = AsyncOpenAI(api_key="k", base_url="http://example.invalid/v1")
    assert cls(client=injected)._client is injected


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_declares_tool_call_support(provider: str):
    """They are the OpenAI provider, so they inherit its capabilities."""
    from actants.llm.openai_compatible import openai_compatible_provider

    cls = openai_compatible_provider(provider, *OPENAI_COMPATIBLE_PROVIDERS[provider])
    built = cls(api_key="k")
    assert built.supports_tool_calls
    assert built.supports_streaming_tools


# --------------------------------------------------------------------------
# Actionable errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
def test_missing_api_key_names_the_env_var(provider: str):
    env_var = _PROVIDER_REQUIREMENTS[provider][0]
    with pytest.raises(MissingAPIKeyError) as exc:
        LLM(provider=provider, model="m")
    msg = str(exc.value)
    assert env_var in msg, "the message must name the exact env var to set"
    assert "LLM()" in msg, "the message must offer the no-API-key local path"


@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("deepsek", "deepseek"),
        ("togethr", "together"),
        ("fireworkz", "fireworks"),
        ("perplexty", "perplexity"),
        ("openroutr", "openrouter"),
    ],
)
def test_typo_suggests_the_right_provider(typo: str, expected: str):
    with pytest.raises(UnknownProviderError) as exc:
        LLM(provider=typo)
    assert f"Did you mean {expected!r}?" in str(exc.value)


def test_unknown_provider_lists_the_new_ones():
    with pytest.raises(UnknownProviderError) as exc:
        LLM(provider="definitely-not-a-provider")
    msg = str(exc.value)
    for provider in NEW_PROVIDERS:
        assert provider in msg, f"{provider!r} missing from the known-providers list"


def test_tool_error_mentions_the_new_providers():
    """The 'use a provider that supports tools' hint must not be stale."""
    from actants.llm.errors import tool_calls_not_supported

    msg = str(tool_calls_not_supported("custom", ["add"]))
    assert "xai" in msg and "deepseek" in msg


# --------------------------------------------------------------------------
# The request actually goes to the right host, with the right shape.
# --------------------------------------------------------------------------


def _stub_openai(base_url: str, captured: dict):
    """An AsyncOpenAI wired to an in-memory transport. No socket is opened."""
    from openai import AsyncOpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": captured["body"]["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    return AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("provider", NEW_PROVIDERS)
async def test_completion_hits_the_provider_host(provider: str):
    from actants.llm.openai_compatible import openai_compatible_provider

    base_url, _ = OPENAI_COMPATIBLE_PROVIDERS[provider]
    captured: dict = {}
    cls = openai_compatible_provider(provider, *OPENAI_COMPATIBLE_PROVIDERS[provider])
    built = cls(client=_stub_openai(base_url, captured))

    result = await built.complete(
        messages=[__import__("actants").ChatMessage(role="user", content="ping")],
        model="test-model",
    )

    assert result.content == "pong"
    assert result.provider == provider
    assert captured["url"].startswith(base_url.rstrip("/"))
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["messages"] == [{"role": "user", "content": "ping"}]


async def test_completion_through_the_llm_client_records_cost_honestly():
    """These hosts have no verified prices, so their cost must read as unknown."""
    from actants import ChatMessage, CostTracker
    from actants.llm.openai_compatible import openai_compatible_provider

    base_url, _ = OPENAI_COMPATIBLE_PROVIDERS["deepseek"]
    captured: dict = {}
    cls = openai_compatible_provider("deepseek", *OPENAI_COMPATIBLE_PROVIDERS["deepseek"])
    tracker = CostTracker()
    llm = LLM(provider=cls(client=_stub_openai(base_url, captured)), cost_tracker=tracker)

    await llm.complete([ChatMessage(role="user", content="ping")], model="deepseek-chat")

    assert tracker.has_untracked_cost
    assert tracker.snapshot()["untracked_models"] == ["deepseek/deepseek-chat"]
