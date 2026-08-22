"""Model-independent generation session state."""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from uuid import uuid4


class GenerationStatus(str, Enum):
    """Lifecycle states for a response generation."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


@dataclass
class GenerationSession:
    """Minimal lifecycle record for one generation request."""

    id: str = field(default_factory=lambda: uuid4().hex)
    status: GenerationStatus = GenerationStatus.CREATED
    created_at: float = field(default_factory=time)
    started_at: float | None = None
    cancelled_at: float | None = None
    completed_at: float | None = None
