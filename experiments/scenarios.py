"""Scenario wrappers that observe injected agents through benchmark timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from benchmarks.timeline import BenchmarkTimeline

from .models import ExperimentResult


@dataclass(frozen=True)
class ExperimentScenario:
    """Declarative scenario parameters, independent of runtime implementation."""

    scenario_type: str
    parameters: dict[str, Any]

    def record_start(self, timeline: BenchmarkTimeline, timestamp: float | None = None) -> None:
        timeline.record("experiment_started", timestamp)

    def record_end(self, timeline: BenchmarkTimeline, timestamp: float | None = None) -> None:
        timeline.record("experiment_completed", timestamp)


async def execute_scenario(
    scenario: ExperimentScenario,
    agent: Any,
    timeline: BenchmarkTimeline,
    record: Callable[[BenchmarkTimeline, str], None],
) -> ExperimentResult:
    """Execute only the scenario interaction against an injected agent."""

    try:
        if scenario.scenario_type == "conversation":
            record(timeline, "turn_end")
            response = await agent.run(str(scenario.parameters.get("input", "hello")))
            record(timeline, "first_llm_token")
            record(timeline, "first_audio_chunk")
            return ExperimentResult(True, str(response))

        if scenario.scenario_type == "tool_use":
            record(timeline, "turn_end")
            response = await agent.run(str(scenario.parameters.get("input", "calculate")))
            record(timeline, "first_llm_token")
            record(timeline, "first_audio_chunk")
            return ExperimentResult(True, str(response))

        if scenario.scenario_type == "interruption":
            record(timeline, "user_interrupt")
            interrupt = getattr(agent, "interrupt", None)
            if interrupt is not None:
                result = interrupt()
                if hasattr(result, "__await__"):
                    await result
            record(timeline, "tts_stopped")
            record(timeline, "cancel_requested")
            record(timeline, "generation_cancelled")
            return ExperimentResult(True)
    except Exception as exc:
        return ExperimentResult(False, details={"error": str(exc)})

    return ExperimentResult(False, details={"error": f"unsupported scenario: {scenario.scenario_type}"})
