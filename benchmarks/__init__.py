"""Declarative benchmark scenarios and evaluation helpers."""

from .evaluator import BenchmarkResult, evaluate
from .scenarios import BenchmarkScenario, load_scenarios

__all__ = [
    "BenchmarkResult",
    "BenchmarkScenario",
    "evaluate",
    "load_scenarios",
]
