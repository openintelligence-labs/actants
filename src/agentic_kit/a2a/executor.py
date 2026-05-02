"""Adapter: agentic-kit Agent → A2A AgentExecutor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue

    from agentic_kit.agents.agent import Agent


def build_executor(agent: Agent) -> Any:
    """Wrap an agentic-kit Agent in an A2A AgentExecutor.

    Imported lazily so the rest of the framework works without ``a2a-sdk``.
    """
    try:
        from a2a.helpers import (
            get_message_text,
            new_task,
            new_text_artifact_update_event,
            new_text_status_update_event,
        )
        from a2a.server.agent_execution import AgentExecutor
        from a2a.types import TaskState
    except ImportError as exc:
        raise ImportError("A2A support requires `pip install agentic-kit[a2a]`") from exc

    class _AgenticKitExecutor(AgentExecutor):  # type: ignore[misc]
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            prompt = get_message_text(context.message) if context.message else ""

            # A2A spec: enqueue the Task itself before any status/artifact updates,
            # so downstream consumers have a Task to attach events to.
            if context.current_task is None:
                await event_queue.enqueue_event(
                    new_task(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        state=TaskState.TASK_STATE_SUBMITTED,
                    )
                )

            if not prompt:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text="empty prompt",
                    )
                )
                return

            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    state=TaskState.TASK_STATE_WORKING,
                    text="",
                )
            )
            try:
                result = await agent.run(prompt)
            except Exception as exc:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text=str(exc),
                    )
                )
                return

            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    name="response",
                    text=result.content,
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="",
                )
            )

        async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
            # agentic-kit Agent loops are not interruptible mid-step yet.
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    state=TaskState.TASK_STATE_CANCELED,
                    text="cancellation not supported",
                )
            )

    return _AgenticKitExecutor()
