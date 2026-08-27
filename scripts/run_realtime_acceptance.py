"""Evaluate realtime acceptance reports with explicit evidence labels."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.metrics import ACCEPTANCE_LABELS, evaluate_acceptance
from benchmarks.real_hardware_runner import HardwareValidationError, RealHardwareBenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or evaluate the realtime acceptance gates with explicit evidence labels."
    )
    parser.add_argument(
        "--label",
        default="model-integration",
        choices=sorted(ACCEPTANCE_LABELS),
        help="Evidence tier to assign to this report.",
    )
    parser.add_argument(
        "--profile",
        default="local_gpu",
        help="Runtime profile to validate or benchmark.",
    )
    parser.add_argument(
        "--metrics-json",
        help="Path to a JSON file containing metrics or a prior report payload.",
    )
    parser.add_argument(
        "--report",
        help="Optional output path for the evaluated acceptance JSON report.",
    )
    parser.add_argument(
        "--audio",
        help="Recorded audio path used for model-integration runs; documented for operator traceability.",
    )
    parser.add_argument(
        "--explicit-real-run",
        action="store_true",
        help="Required for hardware-e2e labeling; confirms this came from a real browser/headset run.",
    )
    parser.add_argument(
        "--skip-runtime-benchmark",
        action="store_true",
        help="Evaluate the supplied metrics JSON without starting runtime validation.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_metrics(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]
    if not isinstance(payload, dict):
        raise ValueError("metrics JSON must be an object or contain a metrics object")
    return {name: float(value) for name, value in payload.items()}


async def _runtime_report(profile: str) -> dict[str, Any]:
    report = await RealHardwareBenchmarkRunner(profile=profile).run()
    return report.to_dict()


def build_acceptance_payload(
    *,
    label: str,
    metrics: dict[str, float],
    explicit_real_run: bool,
    environment: dict[str, Any] | None = None,
    audio: str | None = None,
) -> dict[str, Any]:
    required_measurements_present = all(
        name in metrics
        for name in (
            "duck_latency_ms",
            "interrupt_latency_ms",
            "backchannel_restore_latency_ms",
            "first_token_latency_ms",
            "first_audio_latency_ms",
            "first_translated_audio_latency_ms",
            "stale_output_count",
            "resume_phrase_error_count",
        )
    )
    result = evaluate_acceptance(
        metrics,
        label=label,
        explicit_real_run=explicit_real_run,
        required_measurements_present=required_measurements_present,
    )
    payload = {
        "label": result.label,
        "passed": result.passed,
        "failures": list(result.failures),
        "metrics": result.metrics,
        "explicit_real_run": explicit_real_run,
        "required_measurements_present": required_measurements_present,
        "environment": dict(environment or {}),
    }
    if audio is not None:
        payload["audio"] = audio
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = _load_metrics(args.metrics_json)
    environment: dict[str, Any] = {}
    if not args.skip_runtime_benchmark:
        try:
            report = asyncio.run(_runtime_report(args.profile))
        except HardwareValidationError as exc:
            print(str(exc))
            return 1
        environment = dict(report.get("environment", {}))
        metrics = {**dict(report.get("metrics", {})), **metrics}
    payload = build_acceptance_payload(
        label=args.label,
        metrics=metrics,
        explicit_real_run=bool(args.explicit_real_run),
        environment=environment,
        audio=args.audio,
    )
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
