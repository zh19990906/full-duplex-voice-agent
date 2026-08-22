"""Explicit real-hardware validation and benchmark runner.

This module never substitutes fake devices/models and never invents latency
events. A deployment supplies a session hook that drives the real runtime and
records events on the shared :class:`BenchmarkTimeline`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import resource
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from src.application.entrypoint import ProductionRuntime
from src.config import ConfigLoader

from .metrics import calculate_metrics
from .report import HardwareBenchmarkReport
from .timeline import BenchmarkTimeline


class HardwareValidationError(RuntimeError):
    """Raised when required physical deployment resources are unavailable."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("Hardware validation failed:\n" + "\n".join(f"- {error}" for error in errors))


class HardwareValidationResult:
    """Serializable validation outcome before starting a benchmark."""

    def __init__(
        self,
        errors: tuple[str, ...],
        environment: Mapping[str, Any],
        model_paths: Mapping[str, str],
    ) -> None:
        self.errors = errors
        self.environment = dict(environment)
        self.model_paths = dict(model_paths)

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if not self.valid:
            raise HardwareValidationError(self.errors)


class DefaultHardwareProbe:
    """Probe sounddevice, local model directories, and NVIDIA availability."""

    def _sounddevice(self) -> Any:
        try:
            import sounddevice  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sounddevice package is unavailable") from exc
        return sounddevice

    def check_microphone(self, audio: Mapping[str, Any]) -> bool:
        devices = self._sounddevice().query_devices()
        return any(int(device.get("max_input_channels", 0)) > 0 for device in devices)

    def check_speaker(self, audio: Mapping[str, Any]) -> bool:
        devices = self._sounddevice().query_devices()
        return any(int(device.get("max_output_channels", 0)) > 0 for device in devices)

    def check_model_path(self, path: Path) -> bool:
        return path.is_dir()

    def gpu_info(self, config: Mapping[str, Any]) -> dict[str, Any]:
        if shutil.which("nvidia-smi") is None:
            return {"available": False, "name": None}
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        name = result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None
        return {"available": bool(name), "name": name}


def _model_path(config: Mapping[str, Any], settings: Mapping[str, Any], name: str) -> Path:
    raw_path = settings.get("model_path", settings.get("local_path"))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"missing model path: {name}")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"model path must be relative: {name}")
    models = config.get("models", {})
    root_value = os.environ.get("MODEL_HOME", models.get("model_root", "./models"))
    root = Path(str(root_value)).expanduser()
    if path.parts and path.parts[0] == "models":
        path = Path(*path.parts[1:])
    return (root / path).resolve()


def validate_hardware(
    config: Mapping[str, Any],
    probe: Any | None = None,
) -> HardwareValidationResult:
    """Validate all physical/model requirements without fallback behavior."""

    probe = probe or DefaultHardwareProbe()
    errors: list[str] = []
    audio = config.get("audio", {})
    try:
        microphone = bool(probe.check_microphone(audio))
    except Exception as exc:
        microphone = False
        errors.append(f"microphone probe failed: {exc}")
    try:
        speaker = bool(probe.check_speaker(audio))
    except Exception as exc:
        speaker = False
        errors.append(f"speaker probe failed: {exc}")
    if not microphone:
        errors.append("microphone device unavailable")
    if not speaker:
        errors.append("speaker device unavailable")

    model_paths: dict[str, str] = {}
    models = config.get("models", {})
    for name in ("asr", "llm", "tts"):
        try:
            path = _model_path(config, models[name], name)
            model_paths[name] = str(path)
            if not probe.check_model_path(path):
                errors.append(f"missing model path: {name}")
        except (KeyError, ValueError, OSError) as exc:
            errors.append(str(exc))

    try:
        gpu = dict(probe.gpu_info(config))
    except Exception as exc:
        gpu = {"available": False, "name": None}
        errors.append(f"GPU probe failed: {exc}")
    requires_cuda = any(
        str(models.get(name, {}).get("device", "")).lower().startswith("cuda")
        for name in ("asr", "llm", "tts")
    )
    if requires_cuda and not gpu.get("available"):
        errors.append("GPU unavailable but CUDA model is configured")

    environment = {
        "profile": config.get("runtime", {}).get("environment", "hardware_benchmark"),
        "gpu": gpu.get("name"),
        "microphone": microphone,
        "speaker": speaker,
    }
    return HardwareValidationResult(tuple(errors), environment, model_paths)


SessionHook = Callable[[Any, BenchmarkTimeline], Awaitable[None] | None]
RuntimeFactory = Callable[[str, Mapping[str, Any]], Any]


def _memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(usage / divisor, 6)


class RealHardwareBenchmarkRunner:
    """Run validation and real-session observations on the shared timeline."""

    def __init__(
        self,
        profile: str = "hardware_benchmark",
        config_loader: ConfigLoader | Any | None = None,
        probe: Any | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.profile = profile
        self.config_loader = config_loader or ConfigLoader()
        self.probe = probe or DefaultHardwareProbe()
        self.runtime_factory = runtime_factory or (
            lambda selected_profile, _config: ProductionRuntime(
                profile=selected_profile,
                config_loader=self.config_loader,
            )
        )

    async def run(self, session: SessionHook | None = None) -> HardwareBenchmarkReport:
        config = self.config_loader.load(self.profile)
        validation = validate_hardware(config, self.probe)
        validation.require_valid()
        timeline = BenchmarkTimeline()
        timeline.record("startup_started")
        runtime = self.runtime_factory(self.profile, config)
        initialized = False
        try:
            await runtime.initialize()
            initialized = True
            timeline.record("runtime_ready")
            if session is not None:
                result = session(runtime, timeline)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    await result
        finally:
            if initialized:
                await runtime.shutdown()
        metrics = calculate_metrics(timeline)
        return HardwareBenchmarkReport(
            metrics=metrics,
            environment=validation.environment,
            system={"memory_mb": _memory_mb()},
        )


async def run_real_benchmark(
    profile: str = "hardware_benchmark",
    session: SessionHook | None = None,
) -> HardwareBenchmarkReport:
    """Run the real validation path; latency requires a real session hook."""

    return await RealHardwareBenchmarkRunner(profile=profile).run(session)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and benchmark real voice-agent hardware")
    parser.add_argument("--profile", default="hardware_benchmark")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        report = asyncio.run(run_real_benchmark(profile=args.profile))
    except HardwareValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(report.to_json() if args.as_json else report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
