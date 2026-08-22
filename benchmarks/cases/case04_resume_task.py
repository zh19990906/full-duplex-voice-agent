"""Case 04: resumable task contract placeholder."""

from benchmarks.scenarios import BenchmarkScenario
from src.core.events.events import TaskResumeEvent


SCENARIO = BenchmarkScenario(
    scenario_id="case04",
    description="Declare resuming a counting task from its saved state.",
    input_events=(
        TaskResumeEvent(
            "case04-event",
            4.0,
            "benchmark",
            {"task_type": "counting", "state": {"current_number": 5}},
        ),
    ),
    expected_actions=(),
    validation_rules=("task_state_restored", "resume_from_current_number"),
    metadata={
        "task_type": "counting",
        "current_number": 5,
        "expected_resume_number": 6,
    },
)
