import unittest

from benchmarks.cases.case01_backchannel import SCENARIO as BACKCHANNEL
from benchmarks.cases.case02_interrupt import SCENARIO as INTERRUPT
from benchmarks.cases.case03_translation import SCENARIO as TRANSLATION
from benchmarks.cases.case04_resume_task import SCENARIO as RESUME_TASK
from benchmarks.evaluator import BenchmarkResult, evaluate
from benchmarks.scenarios import load_scenarios
from src.controller.actions import ActionType, ControllerAction
from src.controller.states import ControllerState


class BenchmarkTest(unittest.TestCase):
    def test_scenarios_load_with_all_four_case_ids(self):
        scenarios = load_scenarios()

        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            ["case01", "case02", "case03", "case04"],
        )

    def test_case_definitions_capture_required_contracts(self):
        self.assertEqual(BACKCHANNEL.expected_state, ControllerState.SPEAKING)
        self.assertEqual(
            BACKCHANNEL.expected_actions,
            (ControllerAction(ActionType.CONTINUE_GENERATION),),
        )
        self.assertEqual(INTERRUPT.expected_state, ControllerState.INTERRUPTED)
        self.assertEqual(
            INTERRUPT.expected_actions,
            (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
            ),
        )
        self.assertIn("first_audio_latency", TRANSLATION.validation_rules)
        self.assertEqual(RESUME_TASK.metadata["current_number"], 5)
        self.assertEqual(RESUME_TASK.metadata["expected_resume_number"], 6)

    def test_evaluator_returns_pass_for_matching_state_and_actions(self):
        result = evaluate(
            BACKCHANNEL,
            actual_state=ControllerState.SPEAKING,
            actual_actions=(ControllerAction(ActionType.CONTINUE_GENERATION),),
        )

        self.assertEqual(result, BenchmarkResult(passed=True, errors=[]))

    def test_evaluator_reports_action_and_state_mismatches(self):
        result = evaluate(
            INTERRUPT,
            actual_state=ControllerState.SPEAKING,
            actual_actions=(ControllerAction(ActionType.STOP_RESPONSE),),
        )

        self.assertFalse(result.passed)
        self.assertIn("state", " ".join(result.errors))
        self.assertIn("actions", " ".join(result.errors))

    def test_declarative_placeholder_case_has_no_executable_outcome(self):
        result = evaluate(TRANSLATION)

        self.assertTrue(result.passed)
        self.assertEqual(result.errors, [])
