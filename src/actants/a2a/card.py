from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actants.tools.registry import ToolRegistry


def build_agent_card(
    *,
    name: str,
    description: str,
    version: str,
    url: str,
    tools: ToolRegistry | None = None,
    streaming: bool = True,
    extra_skills: list[Any] | None = None,
) -> Any:
    """Build an A2A AgentCard from actants metadata.

    Each tool in ``tools`` becomes a skill on the card so peer agents can
    discover capabilities. ``url`` is the public base URL where this agent
    is reachable (used as the JSON-RPC endpoint URL).
    """
    try:
        from a2a.types import (
            AgentCapabilities,
            AgentCard,
            AgentInterface,
            AgentSkill,
        )
    except ImportError as exc:
        raise ImportError("A2A support requires `pip install actants[a2a]`") from exc

    skills: list[Any] = []
    if tools is not None:
        for tool in tools.list():
            skills.append(
                AgentSkill(
                    id=tool.name,
                    name=tool.name,
                    description=tool.description,
                    tags=[],
                    examples=[],
                    input_modes=["text/plain"],
                    output_modes=["text/plain"],
                )
            )
    if extra_skills:
        skills.extend(extra_skills)

    # AgentCard always advertises at least one skill — synthesize a default
    # 'chat' skill if the agent exposes no tools, so peers see something.
    if not skills:
        skills.append(
            AgentSkill(
                id="chat",
                name="chat",
                description="Free-form conversation",
                tags=[],
                examples=[],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        )

    return AgentCard(
        name=name,
        description=description,
        version=version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=False,
        ),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=url),
        ],
        skills=skills,
    )
