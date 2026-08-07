from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from actants.tools.base import Tool, ToolError


def RemoteAgent(  # noqa: N802 — factory that returns a Tool, name reads as a class
    base_url: str,
    *,
    name: str | None = None,
    description: str | None = None,
    agent_card_path: str = "/.well-known/agent-card.json",
    timeout: float = 60.0,
) -> Tool:
    """Build a Tool that proxies to a remote A2A agent.

    Construct with the peer's base URL; the Agent Card is resolved lazily on first
    call. Drops directly into a ToolRegistry — the local agent treats it like any
    other tool.

    Example::

        remote = RemoteAgent("https://research-agent.example.com")
        registry = ToolRegistry()
        registry.register(remote)
        agent = Agent(llm=LLM(), tools=registry)
    """
    base_url = base_url.rstrip("/")
    tool_name = name or _slug_from_url(base_url)
    tool_desc = description or f"Send a prompt to remote A2A agent at {base_url}"

    async def handler(prompt: str) -> str:
        return await _call_remote_agent(
            base_url=base_url,
            agent_card_path=agent_card_path,
            timeout=timeout,
            prompt=prompt,
        )

    return Tool(
        name=tool_name,
        description=tool_desc,
        input_schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        handler=handler,
    )


async def _call_remote_agent(
    *, base_url: str, agent_card_path: str, timeout: float, prompt: str
) -> str:
    try:
        import httpx
        from a2a.client import A2ACardResolver, ClientConfig, create_client
        from a2a.helpers import (
            get_artifact_text,
            get_stream_response_text,
            new_text_message,
        )
        from a2a.types import Role, SendMessageRequest
    except ImportError as exc:
        raise ToolError("A2A support requires `pip install actants[a2a]`") from exc

    async with httpx.AsyncClient(timeout=timeout) as http:
        resolver = A2ACardResolver(
            httpx_client=http,
            base_url=base_url,
            agent_card_path=agent_card_path,
        )
        card = await resolver.get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(streaming=True, httpx_client=http),
        )
        try:
            request = SendMessageRequest(
                message=new_text_message(prompt, role=Role.ROLE_USER),
            )
            chunks: list[str] = []
            try:
                async for event in client.send_message(request):
                    text = _extract_text(event, get_artifact_text, get_stream_response_text)
                    if text:
                        chunks.append(text)
            except Exception as exc:
                raise ToolError(f"A2A call to {base_url} failed: {exc}") from exc
            return "".join(chunks)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "remote").replace(".", "_").replace("-", "_")
    return f"a2a__{host}"


def _extract_text(event: Any, get_artifact_text: Any, get_stream_response_text: Any) -> str:
    """Best-effort text extraction from any A2A streaming event."""
    for getter in (get_stream_response_text, get_artifact_text):
        try:
            text = getter(event)
            if text:
                # The getters are passed in untyped (they come from a2a.helpers via a
                # lazy import), so coerce rather than trust them to return str.
                return str(text)
        except Exception:
            continue
    return ""
