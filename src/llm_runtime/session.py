"""LLM generation session lifecycle contract."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4


class GenerationSessionStatus(str, Enum):
    """States of a streaming LLM generation session."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


@dataclass
class GenerationSession:
    """Minimal lifecycle record for one LLM generation request."""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    status: GenerationSessionStatus = GenerationSessionStatus.CREATED
    created_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
