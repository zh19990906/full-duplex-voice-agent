"""Case 03: streaming translation contract placeholder."""

from benchmarks.scenarios import BenchmarkScenario


SCENARIO = BenchmarkScenario(
    scenario_id="case03",
    description="Declare future streaming translation measurements.",
    input_events=(),
    expected_actions=(),
    validation_rules=(
        "translation_pipeline_defined",
        "first_audio_latency",
        "translation_quality",
        "stream_stability",
    ),
    metadata={"implementation": "not implemented in Issue #5"},
)
