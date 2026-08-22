"""Small benchmark runner that observes a supplied timeline."""

from __future__ import annotations

from .metrics import calculate_metrics
from .report import BenchmarkReport
from .timeline import BenchmarkTimeline


def run_benchmark(
    scenario: str = "benchmark",
    timeline: BenchmarkTimeline | None = None,
) -> BenchmarkReport:
    """Calculate a report for ``scenario`` without executing runtime work."""

    timeline = timeline or BenchmarkTimeline()
    return BenchmarkReport(scenario=scenario, metrics=calculate_metrics(timeline))
