"""Case 02: interruption and revision."""

from benchmarks.scenarios import BenchmarkScenario
from src.controller.actions import ActionType, ControllerAction
from src.controller.states import ControllerState
from src.core.events.events import UserInterruptEvent


SCENARIO = BenchmarkScenario(
    scenario_id="case02",
    description="User interruption stops the current response for revision.",
    input_events=(
        UserInterruptEvent(
            "case02-event",
            2.0,
            "benchmark",
            {"text": "等等，我想问上海"},
        ),
    ),
    expected_actions=(
        ControllerAction(ActionType.STOP_RESPONSE),
        ControllerAction(ActionType.CANCEL_GENERATION),
    ),
    validation_rules=("state_equals_expected", "actions_equal_expected"),
    initial_state=ControllerState.SPEAKING,
    expected_state=ControllerState.INTERRUPTED,
)
