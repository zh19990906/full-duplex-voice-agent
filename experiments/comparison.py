"""Comparison execution over existing ExperimentRunner reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import ExperimentConfig
from .profiles.models import ExperimentVariant
from .reports import ComparisonReport
from .runner import ExperimentRunner


class ExperimentComparison:
    """Run several configured variants and aggregate their reports."""

    def __init__(self, runner: ExperimentRunner) -> None:
        self.runner = runner

    async def run(
        self,
        config: ExperimentConfig,
        variants: Mapping[str, Any],
    ) -> ComparisonReport:
        reports = []
        for variant_id, value in variants.items():
            variant: ExperimentVariant | None
            agent: Any
            if isinstance(value, tuple) and len(value) == 2:
                variant, agent = value
            else:
                variant, agent = None, value
            report = await self.runner.run(config, agent, variant=variant)
            if report.variant_id is None:
                report.variant_id = variant_id
            reports.append(report)
        return ComparisonReport(config.experiment_id, config.name, tuple(reports))
