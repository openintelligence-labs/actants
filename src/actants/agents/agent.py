from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from actants.agents.events import (
    AgentRunCompleted,
    AgentStepCompleted,
    AgentTextDelta,
    AgentToolCallCompleted,
    AgentToolCallStarted,
)
from actants.agents.hooks import AgentHooks
from actants.agents.memory import ConversationMemory  # noqa: TC001 — runtime use
from actants.llm.base import (
    ChatMessage,
    CompletionResult,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    UsageDelta,
)
from actants.llm.client import LLM
from actants.tools.registry import ToolRegistry

AgentEvent = (
    AgentTextDelta
    | AgentToolCallStarted
    | AgentToolCallCompleted
    | AgentStepCompleted
    | AgentRunCompleted
)


@dataclass
class AgentStep:
    index: int
    completion: CompletionResult
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    final: CompletionResult
    steps: list[AgentStep]
    messages: list[ChatMessage]

    @property
    def content(self) -> str:
        return self.final.content


class Agent:
    """Stateful tool-calling agent.

    Wraps :class:`LLM` with conversation memory, tool registry, and lifecycle hooks.
    For one-shot tool loops without state, use ``LLM.run_agent`` directly.

    Example::

        agent = Agent(llm=LLM(), tools=registry, system="You are a helpful assistant")
        result = await agent.run("what's the weather?")
        result2 = await agent.run("and tomorrow?")  # remembers context
    """

    def __init__(
        self,
        *,
        llm: LLM,
        tools: ToolRegistry | None = None,
        system: str | None = None,
        memory: ConversationMemory | None = None,
        hooks: AgentHooks | None = None,
        max_steps: int = 6,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationMemory(system=system)
        self.hooks = hooks or AgentHooks()
        self.max_steps = max_steps

    async def run(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        tag: str | None = None,
    ) -> AgentResult:
        """Run one user turn; may dispatch tools across multiple LLM steps."""
        self.memory.add_user(prompt)
        steps: list[AgentStep] = []
        limit = max_steps if max_steps is not None else self.max_steps
        specs = self.tools.as_specs() if self.tools else None

        try:
            for i in range(limit):
                msgs = self.memory.messages()
                if self.hooks.before_step is not None:
                    await self.hooks.before_step(i, msgs)

                completion = await self.llm.complete(
                    msgs,
                    tools=specs,
                    use_cache=False,
                    tag=tag,
                )

                step = AgentStep(index=i, completion=completion, tool_calls=completion.tool_calls)
                steps.append(step)

                if self.hooks.after_step is not None:
                    await self.hooks.after_step(i, completion)

                if not completion.tool_calls:
                    self.memory.add_assistant(completion.content)
                    return AgentResult(
                        final=completion,
                        steps=steps,
                        messages=self.memory.messages(),
                    )

                self.memory.add(
                    ChatMessage(
                        role="assistant",
                        content=completion.content,
                        tool_calls=completion.tool_calls,
                    )
                )
                if self.tools is None:
                    raise RuntimeError(
                        "Model requested tool calls but no ToolRegistry was provided"
                    )
                for call in completion.tool_calls:
                    result = await self.tools.call(call.name, **call.arguments)
                    payload = (
                        json.dumps(result.value, default=str)
                        if result.ok
                        else json.dumps({"error": result.error})
                    )
                    step.tool_results.append(payload)
                    if self.hooks.on_tool_call is not None:
                        await self.hooks.on_tool_call(call, result.value if result.ok else None)
                    self.memory.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))
            raise RuntimeError(f"Agent exceeded max_steps={limit} without producing a final answer")
        except Exception as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise

    async def stream(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        tag: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run one user turn, yielding lifecycle events as they happen.

        Yields, in order across N steps:
          - ``AgentTextDelta(text, step)`` — each model text chunk
          - ``AgentToolCallStarted(call, step)`` — when a tool call is dispatched
          - ``AgentToolCallCompleted(call, value, ok, step)`` — when it returns
          - ``AgentStepCompleted(step, completion)`` — end of one LLM call
          - ``AgentRunCompleted(content, final)`` — final answer (terminal)

        Memory is updated incrementally — at the end of streaming, the agent
        state matches what ``run()`` would produce.
        """
        self.memory.add_user(prompt)
        limit = max_steps if max_steps is not None else self.max_steps
        specs = self.tools.as_specs() if self.tools else None

        provider = self.llm.provider
        model = self.llm.settings.model
        temperature = self.llm.settings.temperature
        cost_tracker = self.llm.cost_tracker

        try:
            for step_idx in range(limit):
                msgs = self.memory.messages()
                if self.hooks.before_step is not None:
                    await self.hooks.before_step(step_idx, msgs)

                step_text: list[str] = []
                step_tool_calls: list[ToolCall] = []
                last_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

                async for event in provider.stream_events(
                    messages=msgs,
                    model=model,
                    temperature=temperature,
                    max_tokens=None,
                    tools=specs,
                ):
                    if isinstance(event, TextDelta):
                        step_text.append(event.text)
                        yield AgentTextDelta(text=event.text, step=step_idx)
                    elif isinstance(event, ToolCallDelta):
                        step_tool_calls.append(event.tool_call)
                    elif isinstance(event, UsageDelta):
                        last_usage = event.usage

                completion = CompletionResult(
                    content="".join(step_text),
                    model=model,
                    provider=provider.name,
                    usage=last_usage,
                    tool_calls=step_tool_calls,
                )
                if cost_tracker is not None:
                    cost_tracker.record(completion, tag=tag)

                if self.hooks.after_step is not None:
                    await self.hooks.after_step(step_idx, completion)
                yield AgentStepCompleted(step=step_idx, completion=completion)

                if not step_tool_calls:
                    self.memory.add_assistant(completion.content)
                    yield AgentRunCompleted(content=completion.content, final=completion)
                    return

                self.memory.add(
                    ChatMessage(
                        role="assistant",
                        content=completion.content,
                        tool_calls=step_tool_calls,
                    )
                )
                if self.tools is None:
                    raise RuntimeError(
                        "Model requested tool calls but no ToolRegistry was provided"
                    )
                for call in step_tool_calls:
                    yield AgentToolCallStarted(call=call, step=step_idx)
                    result = await self.tools.call(call.name, **call.arguments)
                    payload = (
                        json.dumps(result.value, default=str)
                        if result.ok
                        else json.dumps({"error": result.error})
                    )
                    yield AgentToolCallCompleted(
                        call=call,
                        value=result.value if result.ok else result.error,
                        ok=result.ok,
                        step=step_idx,
                    )
                    if self.hooks.on_tool_call is not None:
                        await self.hooks.on_tool_call(call, result.value if result.ok else None)
                    self.memory.add(ChatMessage(role="tool", content=payload, tool_call_id=call.id))

            raise RuntimeError(f"Agent stream exceeded max_steps={limit} without a final answer")
        except Exception as exc:
            if self.hooks.on_error is not None:
                await self.hooks.on_error(exc)
            raise

    def reset(self, *, keep_system: bool = True) -> None:
        self.memory.reset(keep_system=keep_system)
