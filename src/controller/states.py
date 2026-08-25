"""Conversation controller state definitions."""

from enum import Enum

from src.realtime.session_state import (
    ConversationMode,
    FloorState,
    ResponseState,
    SessionState,
)


class ControllerState(str, Enum):
    """States owned by the conversation controller."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


__all__ = [
    "ConversationMode",
    "ControllerState",
    "FloorState",
    "ResponseState",
    "SessionState",
]
