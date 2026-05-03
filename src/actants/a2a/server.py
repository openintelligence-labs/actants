from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actants.a2a.card import build_agent_card
from actants.a2a.executor import build_executor

if TYPE_CHECKING:
    from actants.agents.agent import Agent


def build_app(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "0.1.0",
    base_url: str = "http://127.0.0.1:9000",
) -> Any:
    """Build a Starlette app exposing ``agent`` over A2A.

    Returns an ASGI app you can mount inside a larger application or run with
    any ASGI server (uvicorn, hypercorn). For one-line setup use ``serve()``.

    The app mounts:
      * ``GET /.well-known/agent-card.json`` — discovery
      * JSON-RPC at ``/`` — message/send, message/stream, tasks/get, etc.
    """
    try:
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
        from a2a.server.tasks import InMemoryTaskStore
        from starlette.applications import Starlette
    except ImportError as exc:
        raise ImportError(
            "A2A support requires `pip install actants[a2a]`. Install Starlette too."
        ) from exc

    card = build_agent_card(
        name=name or "actants-agent",
        description=description or "An agent built with actants",
        version=version,
        url=base_url,
        tools=agent.tools,
    )
    executor = build_executor(agent)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes: list[Any] = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    return Starlette(routes=routes)


def serve(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "0.1.0",
    host: str = "127.0.0.1",
    port: int = 9000,
) -> None:
    """Run an A2A server exposing ``agent``. Blocks until the process is killed.

    For embedding in a larger app or testing, use ``build_app`` instead.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("`pip install uvicorn` to use serve()") from exc

    base_url = f"http://{host}:{port}"
    app = build_app(
        agent,
        name=name,
        description=description,
        version=version,
        base_url=base_url,
    )
    uvicorn.run(app, host=host, port=port)
