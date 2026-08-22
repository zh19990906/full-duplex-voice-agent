"""Serializable data contract for incremental translation output."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TranslationChunk:
    """One partial or final source/translation text pair."""

    chunk_id: str
    source_text: str
    translated_text: str
    timestamp: float
    is_final: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, serialization-friendly dictionary."""
        return asdict(self)
