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
    committed_text: str | None = None
    replaces_committed: bool = False

    def __post_init__(self) -> None:
        if type(self.revision_id) is not int or self.revision_id < 0:
            raise ValueError("revision_id must be a nonnegative integer")
        if not isinstance(self.unstable_text, str):
            raise TypeError("unstable_text must be a string")
        if self.committed_text is not None and not isinstance(self.committed_text, str):
            raise TypeError("committed_text must be a string or None")
        if not isinstance(self.replaces_committed, bool):
            raise TypeError("replaces_committed must be a boolean")
        if self.replaces_committed:
            if not self.is_final:
                raise ValueError("only final transcript chunks may replace committed text")
            if self.committed_text is None:
                raise ValueError("replacement chunks require committed_text")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
