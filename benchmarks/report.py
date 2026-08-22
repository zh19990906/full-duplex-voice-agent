"""Machine-readable and human-readable benchmark reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkReport:
    scenario: str
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": self.scenario, "metrics": dict(self.metrics)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def to_text(self) -> str:
        lines = ["Full Duplex Benchmark Report", "", f"Scenario: {self.scenario}"]
        for name, value in self.metrics.items():
            is_latency = name.endswith("_ms")
            label = name.removesuffix("_ms").replace("_", " ").capitalize()
            suffix = f"{value:.2f} ms" if is_latency else f"{value:g}"
            lines.append(f"{label}: {suffix}")
        return "\n".join(lines)
