"""Dataclass-based event contracts, including the V1 realtime envelope."""

from dataclasses import dataclass
from typing import Any, ClassVar

from src.realtime.protocol import RealtimeEnvelope


@dataclass(frozen=True)
class BaseEvent:
    """Common envelope shared by every realtime event."""

    event_id: str
    timestamp: float
    source: str
    payload: dict[str, Any]

    event: ClassVar[str] = "BASE_EVENT"

    def to_dict(self) -> dict[str, Any]:
        """Return the stable wire representation of this event."""
        return {
            "event": self.event,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class UserSpeechStartEvent(BaseEvent):
    event: ClassVar[str] = "USER_SPEECH_START"


@dataclass(frozen=True)
class UserSpeechPartialEvent(BaseEvent):
    event: ClassVar[str] = "USER_SPEECH_PARTIAL"


@dataclass(frozen=True)
class UserBackchannelEvent(BaseEvent):
    event: ClassVar[str] = "USER_BACKCHANNEL"


@dataclass(frozen=True)
class UserInterruptEvent(BaseEvent):
    event: ClassVar[str] = "USER_INTERRUPT"


@dataclass(frozen=True)
class UserTurnEndEvent(BaseEvent):
    event: ClassVar[str] = "USER_TURN_END"


@dataclass(frozen=True)
class AssistantSpeechStartEvent(BaseEvent):
    event: ClassVar[str] = "ASSISTANT_SPEECH_START"


@dataclass(frozen=True)
class AssistantSpeechChunkEvent(BaseEvent):
    event: ClassVar[str] = "ASSISTANT_SPEECH_CHUNK"


@dataclass(frozen=True)
class AssistantSpeechStopEvent(BaseEvent):
    event: ClassVar[str] = "ASSISTANT_SPEECH_STOP"


@dataclass(frozen=True)
class TaskPauseEvent(BaseEvent):
    event: ClassVar[str] = "TASK_PAUSE"


@dataclass(frozen=True)
class TaskResumeEvent(BaseEvent):
    event: ClassVar[str] = "TASK_RESUME"


@dataclass(frozen=True)
class TaskStateUpdateEvent(BaseEvent):
    event: ClassVar[str] = "TASK_STATE_UPDATE"
