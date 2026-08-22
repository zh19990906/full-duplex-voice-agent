"""Timestamp collection for benchmark observations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimelineEntry:
    name: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "timestamp": self.timestamp}


class BenchmarkTimeline:
    """Ordered, serializable event timestamps independent of runtime code."""

    def __init__(self) -> None:
        self._entries: list[TimelineEntry] = []

    def record(self, name: str, timestamp: float | None = None) -> TimelineEntry:
        if not name:
            raise ValueError("timeline event name must not be empty")
        entry = TimelineEntry(name, time.monotonic() if timestamp is None else float(timestamp))
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[TimelineEntry, ...]:
        return tuple(self._entries)

    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]

    def first(self, name: str) -> TimelineEntry | None:
        return next((entry for entry in self._entries if entry.name == name), None)

    def count(self, name: str) -> int:
        return sum(entry.name == name for entry in self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {"events": [entry.to_dict() for entry in self._entries]}
