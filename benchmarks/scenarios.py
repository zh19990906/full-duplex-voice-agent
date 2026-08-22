"""Shared declarative benchmark scenario model."""

from dataclasses import dataclass, field
from typing import Any

from src.controller.actions import ControllerAction
from src.controller.states import ControllerState
from src.core.events.events import BaseEvent


@dataclass(frozen=True)
class BenchmarkScenario:
    """A benchmark description without execution behavior."""

    scenario_id: str
    description: str
    input_events: tuple[BaseEvent, ...]
    expected_actions: tuple[ControllerAction, ...]
    validation_rules: tuple[str, ...]
    initial_state: ControllerState | None = None
    expected_state: ControllerState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Load the four declared benchmark cases in stable order."""
    from .cases.case01_backchannel import SCENARIO as case01
    from .cases.case02_interrupt import SCENARIO as case02
    from .cases.case03_translation import SCENARIO as case03
    from .cases.case04_resume_task import SCENARIO as case04

    return (case01, case02, case03, case04)
