"""Provider-independent context construction."""

from __future__ import annotations

from time import time

from .manager import MemoryManager
from .models import ConversationMessage


class ContextBuilder:
    """Convert stored memory plus current input into structured messages."""

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    def build(
        self,
        session_id: str,
        current_user_message: str | None = None,
    ) -> tuple[ConversationMessage, ...]:
        messages = list(self.memory_manager.get_context(session_id))
        if current_user_message is not None:
            messages.append(
                ConversationMessage(
                    role="user",
                    content=current_user_message,
                    timestamp=time(),
                    metadata={"session_id": session_id},
                )
            )
        return tuple(messages)
