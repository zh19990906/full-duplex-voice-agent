"""Streaming TTS session lifecycle contract."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4


class TTSSessionStatus(str, Enum):
    """States of a streaming TTS session."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"


@dataclass
class TTSSession:
    """Minimal lifecycle record for one TTS stream."""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    status: TTSSessionStatus = TTSSessionStatus.CREATED
    created_at: float = field(default_factory=time)
    started_at: float | None = None
    interrupted_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
