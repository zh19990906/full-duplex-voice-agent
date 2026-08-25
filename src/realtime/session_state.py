"""Orthogonal in-process session state for realtime workflows."""

from dataclasses import dataclass
from enum import Enum


class ConversationMode(str, Enum):
    """Persistent conversation mode."""

    CHAT = "CHAT"
    INTERPRETATION = "INTERPRETATION"


class FloorState(str, Enum):
    """Current speech-floor owner."""

    NONE = "NONE"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    OVERLAP = "OVERLAP"


class ResponseState(str, Enum):
    """Current assistant response lifecycle state."""

    IDLE = "IDLE"
    GENERATING = "GENERATING"
    PLAYING = "PLAYING"
    DUCKED = "DUCKED"
    PAUSED = "PAUSED"
    CANCELLING = "CANCELLING"


@dataclass
class SessionState:
    """Independent session dimensions plus optional assistant-act context."""

    mode: ConversationMode = ConversationMode.CHAT
    floor: FloorState = FloorState.NONE
    response: ResponseState = ResponseState.IDLE
    assistant_act: str | None = None


@dataclass
class ResponseRecord:
    """Minimal identity and lifecycle record for a generated response."""

    response_id: str
    generation_epoch: int
    state: ResponseState = ResponseState.GENERATING
