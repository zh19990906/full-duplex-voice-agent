"""Serializable streaming audio chunk contract."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AudioChunk:
    """One partial or final audio output chunk."""

    chunk_id: str
    audio_data: bytes
    timestamp: float
    is_final: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
