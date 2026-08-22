"""Declarative scenario result contracts for the integration demo."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome of one deterministic integration scenario."""

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


SCENARIO_NAMES = (
    "backchannel_continuation",
    "user_interruption",
    "streaming_translation",
    "task_resume",
)
