"""Serializable streaming transcript data contract."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptChunk:
    """One partial or final transcript update."""

    chunk_id: str
    text: str
    timestamp: float
    is_final: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
