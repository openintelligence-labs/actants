"""Adapter: actants Agent → A2A AgentExecutor."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue

    from actants.agents.agent import Agent


def _ids(context: RequestContext) -> tuple[str, str]:
    """Return this request's (task_id, context_id), generating either if absent.

    `RequestContext` types both as ``str | None`` and only auto-generates them when it
    was built with a request payload -- ``_check_or_generate_task_id`` returns early on
    ``if not self._params``. A context constructed without one therefore carries
    ``None`` for both, and the a2a helpers coerce ``None`` to ``""`` rather than
    rejecting it. That produces a Task with ``id=""`` and status events with
    ``task_id=""``, so every such run correlates to the same empty-string task instead
    of failing loudly. Generating here keeps each run addressable.
    """
    task_id = context.task_id or str(uuid.uuid4())
    context_id = context.context_id or str(uuid.uuid4())
    return task_id, context_id


def build_executor(agent: Agent) -> Any:
    """Wrap an actants Agent in an A2A AgentExecutor.

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
        raise ImportError("A2A support requires `pip install actants[a2a]`") from exc

    class _ActantsExecutor(AgentExecutor):
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            prompt = get_message_text(context.message) if context.message else ""
            task_id, context_id = _ids(context)

            # A2A spec: enqueue the Task itself before any status/artifact updates,
            # so downstream consumers have a Task to attach events to.
            if context.current_task is None:
                await event_queue.enqueue_event(
                    new_task(
                        task_id=task_id,
                        context_id=context_id,
                        state=TaskState.TASK_STATE_SUBMITTED,
                    )
                )

            if not prompt:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=task_id,
                        context_id=context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text="empty prompt",
                    )
                )
                return

            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_WORKING,
                    text="",
                )
            )
            try:
                result = await agent.run(prompt)
            except Exception as exc:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=task_id,
                        context_id=context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text=str(exc),
                    )
                )
                return

            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    name="response",
                    text=result.content,
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="",
                )
            )

        async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
            # actants Agent loops are not interruptible mid-step yet.
            task_id, context_id = _ids(context)
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_CANCELED,
                    text="cancellation not supported",
                )
            )

    return _ActantsExecutor()
