"""Streaming ASR session lifecycle contract."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ASRSessionStatus(str, Enum):
    """States of a streaming ASR session."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass
class ASRSession:
    """Language and lifecycle state for one ASR stream."""

    session_id: str
    language: str
    status: ASRSessionStatus = ASRSessionStatus.CREATED

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
