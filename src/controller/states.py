"""Conversation controller state definitions."""

from enum import Enum


class ControllerState(str, Enum):
    """States owned by the conversation controller."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
