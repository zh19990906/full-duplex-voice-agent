"""Serializable streaming token data contract."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TokenChunk:
    """One partial or final token update."""

    chunk_id: str
    text: str
    timestamp: float
    is_final: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
