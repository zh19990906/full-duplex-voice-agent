"""CLI and programmatic experiment runner."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable
from typing import Any

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline

from .config import load_experiment_config
from .models import ExperimentResult
from .reports import ExperimentReport
from .scenarios import ExperimentScenario, execute_scenario


class ExperimentRunner:
    """Run configured scenarios around an injected existing agent instance."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.clock = clock or time.monotonic

    async def run(self, config: Any, agent: Any | None = None) -> ExperimentReport:
        config.validate()
        timeline = BenchmarkTimeline()
        record = self._recorder
        record(timeline, "startup_started")
        if agent is None:
            return ExperimentReport(
                config.experiment_id,
                config.name,
                config.scenario,
                False,
                {},
                {"error": "an existing agent instance must be injected"},
            )
        record(timeline, "runtime_ready")
        scenario = ExperimentScenario(config.scenario, config.parameters)
        scenario.record_start(timeline, self.clock())
        result: ExperimentResult = await execute_scenario(scenario, agent, timeline, record)
        scenario.record_end(timeline, self.clock())
        metrics = calculate_metrics(timeline)
        agent_metrics = self._agent_metrics(agent, result)
        return ExperimentReport(
            config.experiment_id,
            config.name,
            config.scenario,
            result.success,
            metrics,
            agent_metrics,
        )

    def _recorder(self, timeline: BenchmarkTimeline, name: str) -> None:
        timeline.record(name, self.clock())

    @staticmethod
    def _agent_metrics(agent: Any, result: ExperimentResult) -> dict[str, Any]:
        state = getattr(agent, "state", None)
        return {
            "success": result.success,
            "iterations": int(getattr(state, "iteration", 0)),
            "tool_calls": int(getattr(agent, "tool_calls", 0)),
            **result.details,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a configured voice-agent experiment")
    parser.add_argument("experiment_id")
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config, args.experiment_id)
    report = asyncio.run(ExperimentRunner().run(config))
    print(report.to_text())
    print("\nJSON:")
    print(report.to_json())
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
