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
    revision_id: int = 0
    unstable_text: str = ""

    def __post_init__(self) -> None:
        if type(self.revision_id) is not int or self.revision_id < 0:
            raise ValueError("revision_id must be a nonnegative integer")
        if not isinstance(self.unstable_text, str):
            raise TypeError("unstable_text must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
