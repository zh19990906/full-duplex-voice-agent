"""Case 01: backchannel continuation."""

from benchmarks.scenarios import BenchmarkScenario
from src.controller.actions import ActionType, ControllerAction
from src.controller.states import ControllerState
from src.core.events.events import UserBackchannelEvent


SCENARIO = BenchmarkScenario(
    scenario_id="case01",
    description="User backchannel keeps an assistant response speaking.",
    input_events=(
        UserBackchannelEvent("case01-event", 1.0, "benchmark", {"text": "嗯嗯"}),
    ),
    expected_actions=(ControllerAction(ActionType.CONTINUE_GENERATION),),
    validation_rules=("state_equals_expected", "actions_equal_expected"),
    initial_state=ControllerState.SPEAKING,
    expected_state=ControllerState.SPEAKING,
)
