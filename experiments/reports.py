"""Experiment report formats built on existing benchmark metrics."""

from __future__ import annotations

import json
from typing import Any, Mapping

from benchmarks.report import BenchmarkReport


class ExperimentReport:
    """Stable JSON and human-readable representation of one experiment run."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        scenario: str,
        success: bool,
        metrics: Mapping[str, Any],
        agent: Mapping[str, Any],
        variant_id: str | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.scenario = scenario
        self.success = bool(success)
        self.metrics = dict(metrics)
        self.agent = dict(agent)
        self.variant_id = variant_id
        self.benchmark_report = BenchmarkReport(
            scenario=scenario,
            metrics={name: value for name, value in self.metrics.items() if isinstance(value, (int, float))},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": dict(self.agent),
            "experiment_id": self.experiment_id,
            "metrics": dict(self.metrics),
            "name": self.name,
            "scenario": self.scenario,
            "success": self.success,
            **({"variant_id": self.variant_id} if self.variant_id is not None else {}),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def to_text(self) -> str:
        lines = [
            "Experiment Report",
            "",
            f"Name: {self.name}",
            f"Scenario: {self.scenario}",
            f"Success: {str(self.success).lower()}",
            "",
            "Metrics:",
        ]
        if self.variant_id is not None:
            lines.insert(4, f"Variant: {self.variant_id}")
        for name, value in self.metrics.items():
            label = name.removesuffix("_ms").replace("_", " ").capitalize()
            suffix = f"{value:.2f} ms" if name.endswith("_ms") else str(value)
            lines.append(f"{label}: {suffix}")
        return "\n".join(lines)


class ComparisonReport:
    """Human and machine-readable collection of variant reports."""

    def __init__(self, experiment_id: str, name: str, reports: tuple[ExperimentReport, ...]) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.reports = reports

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "variants": [
                {
                    "variant_id": report.variant_id,
                    "success": report.success,
                    "metrics": dict(report.metrics),
                    "agent": dict(report.agent),
                }
                for report in self.reports
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def to_text(self) -> str:
        lines = ["Model Comparison", "", f"Name: {self.name}", ""]
        for report in self.reports:
            lines.append(f"Variant: {report.variant_id}")
            for name, value in report.metrics.items():
                label = name.removesuffix("_ms").replace("_", " ").capitalize()
                suffix = f"{value:.2f} ms" if name.endswith("_ms") else str(value)
                lines.append(f"{label}: {suffix}")
            lines.append("")
        return "\n".join(lines).rstrip()
