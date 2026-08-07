"""The full description of a request, as far as caching is concerned.

Every cache backend — exact-match or semantic — keys off one of these. The point of
having a single object rather than a handful of positional parameters is that adding a
new field that changes the answer becomes a change to *one* type, and every backend
picks it up. The previous design passed ``(messages, model, temperature)`` to the
semantic backend and a longer list to the exact-match backend, so the two disagreed
about what made a request unique and the semantic cache collided on ``max_tokens``,
provider, tools, and response format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actants.llm.base import ChatMessage, ToolSpec

#: Bump when the hashed payload layout changes, so entries written by an older actants
#: can never be misread as current ones. This value is embedded in both the exact-match
#: key and the semantic backend's scope hash, and it is what the on-disk schema version
#: check in :class:`~actants.cache.semantic.SqliteVecCache` compares against.
KEY_VERSION = 3


@dataclass(frozen=True)
class CacheRequest:
    """Everything about a request that can change the answer.

    Anything omitted here becomes a cache collision — two different requests silently
    sharing one answer — so err on the side of including a field.

    Attributes:
        messages: The full message list, including tool-call structure.
        model: Model name as sent to the provider.
        temperature: Sampling temperature.
        provider: Provider name (``ollama``, ``openai``, ...). Two providers serving the
            same model name do not necessarily return the same answer.
        max_tokens: Output-length cap. A 16-token answer and a 4096-token answer to the
            same prompt are different answers.
        tools: Tool definitions offered to the model.
        response_format: Provider-specific structured-output request (e.g. a JSON-schema
            block). Changes the shape of the answer, so it changes the key.
        extra: Provider-specific parameters not modelled above — ``seed``, ``top_p``,
            ``stop``. Populated from the passthrough keyword arguments of
            :meth:`~actants.llm.client.LLM.complete`, and included in the key verbatim,
            so a ``seed=1`` answer is never served to a ``seed=2`` request.
    """

    messages: list[ChatMessage]
    model: str
    temperature: float
    provider: str | None = None
    max_tokens: int | None = None
    tools: list[ToolSpec] | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Hash of every field, including the message contents.

        This is the exact-match cache key: two requests share it only if they are
        identical in every respect.
        """
        return _sha256_json(self._payload(include_messages=True))

    def scope_hash(self) -> str:
        """Hash of every field *except* the message contents.

        Semantic caches match message content by embedding distance rather than by
        equality, so they need a discriminator for everything else — model, provider,
        temperature, ``max_tokens``, tools, response format. Two requests may share an
        answer only if their scope hashes are equal *and* their embeddings are close.

        Message *structure* that is not carried in the embedding — the number of
        messages and their roles — is included here, so that a 1-message and a
        5-message conversation which happen to flatten to similar text cannot collide.
        """
        payload = self._payload(include_messages=False)
        payload["shape"] = [m.role for m in self.messages]
        return _sha256_json(payload)

    def _payload(self, *, include_messages: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "v": KEY_VERSION,
            "provider": self.provider,
            "model": self.model,
            # Round to the precision providers actually honour, but keep enough digits
            # that distinct temperatures stay distinct.
            "temperature": round(float(self.temperature), 6),
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "extra": self.extra,
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in (self.tools or [])
            ],
        }
        if include_messages:
            payload["messages"] = [
                {
                    "role": m.role,
                    "content": m.content,
                    "name": m.name,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ],
                }
                for m in self.messages
            ]
        return payload

    def embedding_text(self) -> str:
        """Flatten the messages into the text a semantic backend embeds."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


def _sha256_json(payload: dict[str, Any]) -> str:
    """Hash canonical JSON rather than concatenated bytes.

    Concatenation lets message content forge a field boundary — a user message
    containing the separator byte could impersonate a different message list. JSON
    escaping makes that impossible.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
