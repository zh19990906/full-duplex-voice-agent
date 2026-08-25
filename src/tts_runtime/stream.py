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
    request_id: str | None = None
    response_id: str | None = None
    generation_epoch: int | None = None
    segment_id: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("request_id", self.request_id), ("response_id", self.response_id)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        for name, value in (("generation_epoch", self.generation_epoch), ("segment_id", self.segment_id)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer or None")
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
