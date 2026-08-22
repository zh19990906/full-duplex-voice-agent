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


@dataclass(frozen=True)
class HardwareBenchmarkReport:
    """Report format for measurements collected from real deployment hooks."""

    metrics: Mapping[str, float]
    environment: Mapping[str, Any]
    system: Mapping[str, Any]
    scenario: str = "real_hardware"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "system": dict(self.system),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def to_text(self) -> str:
        labels = {
            "first_transcript_latency_ms": "ASR First Transcript",
            "first_token_latency_ms": "LLM TTFT",
            "first_audio_latency_ms": "TTS TTFA",
            "interrupt_latency_ms": "Interrupt Latency",
            "cancellation_latency_ms": "Cancellation Latency",
            "cancel_latency_ms": "Cancellation Latency",
            "startup_latency_ms": "Startup Time",
        }
        lines = ["Real Hardware Benchmark Report", "", "Environment:"]
        for name, value in self.environment.items():
            lines.append(f"{name.replace('_', ' ').capitalize()}: {value}")
        lines.append("")
        for name, value in self.metrics.items():
            label = labels.get(name, name.replace("_", " ").capitalize())
            suffix = f"{value:.2f} ms" if name.endswith("_ms") else f"{value:g}"
            lines.append(f"{label}: {suffix}")
        for name, value in self.system.items():
            label = name.replace("_", " ").capitalize()
            suffix = f"{value:.2f} MB" if name.endswith("_mb") else str(value)
            lines.append(f"{label}: {suffix}")
        return "\n".join(lines)
