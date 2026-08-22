"""Replaceable persistence boundary for conversation memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

from .models import ConversationMessage, ConversationState


class MemoryStore(ABC):
    """Minimal storage contract independent of persistence technology."""

    @abstractmethod
    def append_message(self, session_id: str, message: ConversationMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, session_id: str) -> tuple[ConversationMessage, ...]:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, state: ConversationState) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self, session_id: str) -> ConversationState:
        raise NotImplementedError

    @abstractmethod
    def clear_messages(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """Process-local store for development and test deployments."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ConversationMessage]] = defaultdict(list)
        self._states: dict[str, ConversationState] = {}

    def append_message(self, session_id: str, message: ConversationMessage) -> None:
        self._require_state(session_id)
        self._messages[session_id].append(message)

    def get_messages(self, session_id: str) -> tuple[ConversationMessage, ...]:
        self._require_state(session_id)
        return tuple(self._messages[session_id])

    def save_state(self, state: ConversationState) -> None:
        self._states[state.session_id] = state
        self._messages.setdefault(state.session_id, [])

    def get_state(self, session_id: str) -> ConversationState:
        try:
            return self._states[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown memory session: {session_id}") from exc

    def clear_messages(self, session_id: str) -> None:
        self._require_state(session_id)
        self._messages[session_id].clear()

    def delete_session(self, session_id: str) -> None:
        self._states.pop(session_id, None)
        self._messages.pop(session_id, None)

    def _require_state(self, session_id: str) -> ConversationState:
        return self.get_state(session_id)
