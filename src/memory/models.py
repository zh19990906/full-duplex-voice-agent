"""Serializable conversation memory data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Mapping


MESSAGE_ROLES = frozenset({"user", "assistant", "system"})


@dataclass
class ConversationMessage:
    """One provider-independent message in a conversation."""

    role: str
    content: str
    timestamp: float = field(default_factory=time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in MESSAGE_ROLES:
            raise ValueError(f"unsupported conversation message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("conversation message content must be a string")
        self.timestamp = float(self.timestamp)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationState:
    """Long-lived state associated with one conversation session."""

    session_id: str
    summary: str = ""
    user_preferences: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("conversation session_id must not be empty")
        self.user_preferences = dict(self.user_preferences)
        self.current_state = dict(self.current_state)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def message_from_mapping(value: Mapping[str, Any]) -> ConversationMessage:
    """Convert a store mapping into the stable message model."""

    return ConversationMessage(
        role=str(value["role"]),
        content=str(value["content"]),
        timestamp=float(value.get("timestamp", time())),
        metadata=dict(value.get("metadata", {})),
    )
