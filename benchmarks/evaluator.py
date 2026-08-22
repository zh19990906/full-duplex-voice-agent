"""Pure evaluator for declarative controller benchmark outcomes."""

from dataclasses import dataclass, field
from typing import Sequence

from src.controller.actions import ControllerAction
from src.controller.states import ControllerState

from .scenarios import BenchmarkScenario


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of comparing actual values with a benchmark scenario."""

    passed: bool
    errors: list[str] = field(default_factory=list)


def evaluate(
    scenario: BenchmarkScenario,
    actual_state: ControllerState | None = None,
    actual_actions: Sequence[ControllerAction] = (),
) -> BenchmarkResult:
    """Compare only declared state and action expectations."""
    errors: list[str] = []
    if scenario.expected_state is not None and actual_state != scenario.expected_state:
        errors.append(
            f"state mismatch: expected {scenario.expected_state}, got {actual_state}"
        )
    if scenario.expected_actions and tuple(actual_actions) != scenario.expected_actions:
        errors.append(
            "actions mismatch: "
            f"expected {scenario.expected_actions}, got {tuple(actual_actions)}"
        )
    return BenchmarkResult(passed=not errors, errors=errors)
