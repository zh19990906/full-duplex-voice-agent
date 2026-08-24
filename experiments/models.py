"""Experiment execution result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    """Scenario-level result before metric/report aggregation."""

    success: bool
    response: str = ""
    details: dict[str, Any] = field(default_factory=dict)
