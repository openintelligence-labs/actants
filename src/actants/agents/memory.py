from __future__ import annotations

from collections.abc import Iterable

from actants.llm.base import ChatMessage


class ConversationMemory:
    """Append-only chat history with optional truncation by message count.

    Holds ChatMessage objects exactly as they flow through the LLM client. ``max_messages``
    keeps the most recent N (system messages always preserved). For token-aware
    truncation, callers can subclass and override ``trim``.
    """

    def __init__(
        self,
        *,
        system: str | None = None,
        max_messages: int | None = None,
    ) -> None:
        self._messages: list[ChatMessage] = []
        if system is not None:
            self._messages.append(ChatMessage(role="system", content=system))
        self._max_messages = max_messages

    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self.trim()

    def add_user(self, content: str) -> None:
        self.add(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.add(ChatMessage(role="assistant", content=content))

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        for m in messages:
            self.add(m)

    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def reset(self, *, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [m for m in self._messages if m.role == "system"]
        else:
            self._messages = []

    def trim(self) -> None:
        if self._max_messages is None or len(self._messages) <= self._max_messages:
            return
        system = [m for m in self._messages if m.role == "system"]
        non_system = [m for m in self._messages if m.role != "system"]
        keep = self._max_messages - len(system)
        if keep < 0:
            keep = 0
        self._messages = system + non_system[-keep:]

    def __len__(self) -> int:
        return len(self._messages)
