"""Audio frame data contract."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AudioFrame:
    """One transport-ready audio frame without processing semantics."""

    frame_id: str
    timestamp: float
    sample_rate: int
    channels: int
    data: bytes

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary for transport serialization."""
        return asdict(self)
