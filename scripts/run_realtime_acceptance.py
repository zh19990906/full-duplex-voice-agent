"""Run or evaluate realtime acceptance with evidence-bound provenance."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.metrics import (
    ACCEPTANCE_LABELS,
    AcceptanceProvenance,
    evaluate_acceptance,
)
from benchmarks.real_hardware_runner import (
    AcceptanceEvidence,
    BrowserHeadsetAcceptanceDriver,
    RecordedAudioRealtimeDriver,
)


IMPORTED_LABELS = frozenset({"unit", "simulated"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or evaluate realtime acceptance with evidence-bound labels."
    )
    parser.add_argument(
        "--label",
        default="model-integration",
        choices=sorted(ACCEPTANCE_LABELS),
        help="Evidence tier to evaluate.",
    )
    parser.add_argument(
        "--profile",
        default="local_gpu",
        help="Runtime profile used by a live capture driver.",
    )
    parser.add_argument(
        "--metrics-json",
        help="Imported metrics JSON; valid only for unit or simulated evidence.",
    )
    parser.add_argument(
        "--report",
        help="Optional output path for the atomically written acceptance report.",
    )
    parser.add_argument(
        "--audio",
        help="PCM16 mono 16 kHz WAV driven through the Task 13 realtime session.",
    )
    parser.add_argument(
        "--browser-headset-driver",
        help=(
            "Executable capture driver for hardware-e2e. It receives "
            "--profile and must print captured timeline/metrics JSON."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_metrics(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]
    if not isinstance(payload, dict):
        raise ValueError("metrics JSON must be an object or contain a metrics object")
    return {name: float(value) for name, value in payload.items()}


def build_acceptance_payload(
    *,
    label: str,
    metrics: dict[str, float],
    provenance: AcceptanceProvenance | None,
    environment: dict[str, Any] | None = None,
    audio: str | None = None,
    timeline: list[dict[str, Any]] | None = None,
    _evidence_capability: object | None = None,
) -> dict[str, Any]:
    result = evaluate_acceptance(
        metrics,
        label=label,
        provenance=provenance,
        _evidence_capability=_evidence_capability,
    )
    payload: dict[str, Any] = {
        "label": result.label,
        "passed": result.passed,
        "failures": list(result.failures),
        "metrics": result.metrics,
        "provenance": provenance.to_dict() if provenance is not None else None,
        "environment": dict(environment or {}),
    }
    if audio is not None:
        payload["audio"] = audio
    if timeline is not None:
        payload["timeline"] = list(timeline)
    return payload


def _command_capture(command: str):
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("browser/headset capture driver command must not be empty")

    async def capture(profile: str) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            "--profile",
            profile,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"browser/headset capture driver exited {process.returncode}: {detail}"
            )
        payload = json.loads(stdout.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("browser/headset capture driver output must be a JSON object")
        return payload

    return capture


async def run_acceptance(
    args: argparse.Namespace,
    *,
    recorded_audio_driver: Any | None = None,
    browser_headset_driver: Any | None = None,
) -> dict[str, Any]:
    """Resolve the label's permitted evidence path and evaluate its metrics."""

    label = str(args.label)
    metrics_json = getattr(args, "metrics_json", None)
    audio = getattr(args, "audio", None)
    driver_command = getattr(args, "browser_headset_driver", None)

    if metrics_json is not None and label not in IMPORTED_LABELS:
        raise ValueError("imported metrics JSON is only unit or simulated evidence")

    if label in IMPORTED_LABELS:
        if metrics_json is None:
            raise ValueError(f"{label} evaluation requires --metrics-json")
        if audio is not None or driver_command is not None:
            raise ValueError(f"{label} evaluation cannot run a live capture driver")
        return build_acceptance_payload(
            label=label,
            metrics=_load_metrics(metrics_json),
            provenance=None,
        )

    if label == "model-integration":
        if audio is None:
            raise ValueError("model-integration requires --audio")
        if driver_command is not None:
            raise ValueError("model-integration cannot use a browser/headset capture driver")
        driver = recorded_audio_driver or RecordedAudioRealtimeDriver()
        evidence = await driver.run(Path(audio), profile=args.profile)
        return _payload_from_evidence(label, evidence, audio=str(Path(audio)))

    if audio is not None:
        raise ValueError("hardware-e2e must use a browser/headset capture driver, not --audio")
    driver = browser_headset_driver
    if driver is None and driver_command is not None:
        driver = BrowserHeadsetAcceptanceDriver(_command_capture(driver_command))
    if driver is None:
        raise ValueError("hardware-e2e requires a real browser/headset capture driver")
    evidence = await driver.run(profile=args.profile)
    return _payload_from_evidence(label, evidence)


def _payload_from_evidence(
    label: str,
    evidence: AcceptanceEvidence,
    *,
    audio: str | None = None,
) -> dict[str, Any]:
    if not isinstance(evidence, AcceptanceEvidence):
        raise TypeError("acceptance driver must return AcceptanceEvidence")
    return build_acceptance_payload(
        label=label,
        metrics=evidence.metrics,
        provenance=evidence.provenance,
        environment=evidence.environment,
        audio=audio,
        timeline=[entry.to_dict() for entry in evidence.timeline.entries],
        _evidence_capability=evidence._capability,
    )


def write_report_atomic(path: str | Path, text: str) -> None:
    """Replace a report only after its complete contents reach a sibling file."""

    target = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = asyncio.run(run_acceptance(args))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"acceptance error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report:
        write_report_atomic(args.report, text)
    print(text, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
