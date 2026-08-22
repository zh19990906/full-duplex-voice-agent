"""Translation session lifecycle contract."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TranslationSessionStatus(str, Enum):
    """States of a translation session."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass
class TranslationSession:
    """Language pair and lifecycle state for one translation stream."""

    session_id: str
    source_language: str
    target_language: str
    status: TranslationSessionStatus = TranslationSessionStatus.CREATED

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
