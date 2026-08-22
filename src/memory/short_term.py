"""Bounded recent-message memory."""

from __future__ import annotations

from .models import ConversationMessage
from .store import MemoryStore


class ShortTermMemory:
    """Maintain ordered recent messages for one session."""

    def __init__(self, store: MemoryStore, session_id: str, max_messages: int = 20) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self.store = store
        self.session_id = session_id
        self.max_messages = max_messages

    def append(self, message: ConversationMessage) -> None:
        self.store.append_message(self.session_id, message)
        messages = self.store.get_messages(self.session_id)
        if len(messages) > self.max_messages:
            retained = messages[-self.max_messages :]
            self.store.clear_messages(self.session_id)
            for item in retained:
                self.store.append_message(self.session_id, item)

    def recent(self) -> tuple[ConversationMessage, ...]:
        return self.store.get_messages(self.session_id)[-self.max_messages :]

    def clear(self) -> None:
        self.store.clear_messages(self.session_id)
