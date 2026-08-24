"""CLI and programmatic experiment runner."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable
from typing import Any

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline

from .config import ExperimentConfigError, load_experiment_config
from .models import ExperimentResult
from .reports import ExperimentReport
from .scenarios import ExperimentScenario, execute_scenario
from .profiles.loader import ExperimentProfileBundle, ExperimentRuntimeContext
from .profiles.models import ExperimentVariant


class ExperimentRunner:
    """Run configured scenarios around an injected existing agent instance."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        agent_factory: Callable[[ExperimentRuntimeContext], Any] | None = None,
        profile_bundle: ExperimentProfileBundle | None = None,
    ) -> None:
        self.clock = clock or time.monotonic
        self.agent_factory = agent_factory
        self.profile_bundle = profile_bundle

    async def run(
        self,
        config: Any,
        agent: Any | None = None,
        variant: str | ExperimentVariant | None = None,
    ) -> ExperimentReport:
        config.validate()
        context = None
        if variant is not None:
            if self.profile_bundle is not None:
                context = self.profile_bundle.context(variant)
            elif isinstance(variant, str) or self.agent_factory is not None:
                raise ValueError("profile_bundle is required to resolve a named variant or create an agent")
        timeline = BenchmarkTimeline()
        record = self._recorder
        record(timeline, "startup_started")
        if agent is None and self.agent_factory is not None:
            if context is None:
                raise ValueError("variant is required when using agent_factory")
            agent = self.agent_factory(context)
        if agent is None:
            return ExperimentReport(
                config.experiment_id,
                config.name,
                config.scenario,
                False,
                {},
                {"error": "an existing agent instance must be injected"},
                variant_id=(context.variant.variant_id if context is not None else getattr(variant, "variant_id", None)),
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
            variant_id=(context.variant.variant_id if context is not None else getattr(variant, "variant_id", None)),
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
    parser.add_argument("experiment_id", nargs="?")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--profile-config", default="configs/experiment_profiles.yaml")
    args = parser.parse_args(argv)

    if args.compare:
        from .comparison import ExperimentComparison

        bundle = _load_profile_bundle(args.profile_config)
        config = load_experiment_config(args.config, "conversation_test")
        variants = {name: (bundle.variant(name), None) for name in args.compare}
        report = asyncio.run(ExperimentComparison(ExperimentRunner(profile_bundle=bundle)).run(config, variants))
        print(report.to_text())
        print("\nJSON:")
        print(report.to_json())
        return 0 if all(item.success for item in report.reports) else 1

    if not args.experiment_id:
        parser.error("an experiment id or --compare VARIANT_A VARIANT_B is required")
    try:
        config = load_experiment_config(args.config, args.experiment_id)
        report = asyncio.run(ExperimentRunner().run(config))
    except ExperimentConfigError:
        bundle = _load_profile_bundle(args.profile_config)
        config = load_experiment_config(args.config, "conversation_test")
        report = asyncio.run(ExperimentRunner(profile_bundle=bundle).run(config, variant=args.experiment_id))
    print(report.to_text())
    print("\nJSON:")
    print(report.to_json())
    return 0 if report.success else 1


def _load_profile_bundle(path: str) -> ExperimentProfileBundle:
    """Load the repository's profile documents without loading any model."""

    return ExperimentProfileBundle.from_files(
        path,
        "experiments/profiles/models.yaml",
        "experiments/profiles/prompts.yaml",
        "experiments/profiles/strategies.yaml",
    )


if __name__ == "__main__":
    raise SystemExit(main())
