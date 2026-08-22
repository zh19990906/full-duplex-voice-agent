"""Serializable runtime status snapshots."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeStatus:
    """Immutable operational snapshot with monotonic uptime calculation."""

    profile: str
    runtime_state: str
    loaded_models: Mapping[str, str]
    started_at: float | None = None
    now: float | None = None

    @property
    def uptime_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        current = time.monotonic() if self.now is None else self.now
        return max(0.0, round(current - self.started_at, 6))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "runtime_state": self.runtime_state,
            "loaded_models": dict(self.loaded_models),
            "started_at": self.started_at,
            "uptime_seconds": self.uptime_seconds,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


__all__ = ["RuntimeStatus"]
