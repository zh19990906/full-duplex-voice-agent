"""Serializable API response models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ApiResponse:
    """Transport-neutral HTTP response returned by the API facade."""

    status_code: int
    body: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
