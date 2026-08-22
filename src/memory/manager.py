"""Conversation memory lifecycle management."""

from __future__ import annotations

from collections.abc import Mapping
from time import time
from typing import Any

from .models import ConversationMessage, ConversationState
from .short_term import ShortTermMemory
from .store import InMemoryStore, MemoryStore


class MemoryManager:
    """Own session state and delegate storage to a replaceable store."""

    def __init__(self, store: MemoryStore | None = None, max_messages: int = 20) -> None:
        self.store = store or InMemoryStore()
        self.max_messages = max_messages
        self._short_term: dict[str, ShortTermMemory] = {}

    def create_session(self, session_id: str) -> ConversationState:
        try:
            self.store.get_state(session_id)
        except KeyError:
            pass
        else:
            raise ValueError(f"memory session already exists: {session_id}")
        state = ConversationState(session_id=session_id)
        self.store.save_state(state)
        self._short_term[session_id] = ShortTermMemory(self.store, session_id, self.max_messages)
        return state

    def get_session(self, session_id: str) -> ConversationState:
        return self.store.get_state(session_id)

    def save_state(self, state: ConversationState) -> None:
        self._require_session(state.session_id)
        self.store.save_state(state)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConversationMessage:
        short_term = self._require_session(session_id)
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=time() if timestamp is None else timestamp,
            metadata=dict(metadata or {}),
        )
        short_term.append(message)
        return message

    def get_context(self, session_id: str) -> tuple[ConversationMessage, ...]:
        return self._require_session(session_id).recent()

    def clear(self, session_id: str) -> None:
        self._require_session(session_id).clear()

    def delete_session(self, session_id: str) -> None:
        self._require_session(session_id)
        self.store.delete_session(session_id)
        self._short_term.pop(session_id, None)

    def _require_session(self, session_id: str) -> ShortTermMemory:
        short_term = self._short_term.get(session_id)
        if short_term is not None:
            return short_term
        try:
            self.store.get_state(session_id)
        except KeyError as exc:
            raise KeyError(f"unknown memory session: {session_id}") from exc
        short_term = ShortTermMemory(self.store, session_id, self.max_messages)
        self._short_term[session_id] = short_term
        return short_term
