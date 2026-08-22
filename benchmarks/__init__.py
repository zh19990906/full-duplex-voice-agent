"""Declarative benchmark scenarios and evaluation helpers."""

from .evaluator import BenchmarkResult, evaluate
from .metrics import calculate_metrics, latency_ms
from .report import BenchmarkReport, HardwareBenchmarkReport
from .runner import run_benchmark
from .scenarios import BenchmarkScenario, load_scenarios
from .timeline import BenchmarkTimeline

__all__ = [
    "BenchmarkResult",
    "BenchmarkScenario",
    "evaluate",
    "BenchmarkReport",
    "HardwareBenchmarkReport",
    "BenchmarkTimeline",
    "calculate_metrics",
    "latency_ms",
    "load_scenarios",
    "run_benchmark",
]
