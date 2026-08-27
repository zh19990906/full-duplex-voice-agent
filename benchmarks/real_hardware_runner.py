"""Explicit real-hardware validation and benchmark runner.

This module never substitutes fake devices/models and never invents latency
events. A deployment supplies a session hook that drives the real runtime and
records events on the shared :class:`BenchmarkTimeline`.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import inspect
import os
import resource
import shutil
import subprocess
import sys
import time
import uuid
import wave
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from src.application.entrypoint import ProductionRuntime
from src.config import ConfigLoader
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import (
    AudioFrameHeader,
    PCM16_FRAME_BYTES,
    PCM16_FRAME_SAMPLES,
    PCM16_MONO_CHANNELS,
    PCM16_SAMPLE_RATE,
)

from .metrics import (
    AcceptanceProvenance,
    EvidenceSource,
    METRIC_EVENTS,
    calculate_metrics,
)
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


@dataclass(frozen=True)
class AcceptanceEvidence:
    """Metrics and runner provenance captured by one acceptance driver."""

    metrics: dict[str, float]
    environment: dict[str, Any]
    provenance: AcceptanceProvenance
    timeline: BenchmarkTimeline

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "environment": dict(self.environment),
            "provenance": self.provenance.to_dict(),
            "timeline": [entry.to_dict() for entry in self.timeline.entries],
        }


AcceptanceRuntimeFactory = Callable[[str], Any]
BrowserCapture = Callable[[str], Awaitable[Mapping[str, Any]] | Mapping[str, Any]]
_KNOWN_CAPTURE_EVENTS = {
    event_name
    for interval in METRIC_EVENTS.values()
    for event_name in interval
} | {"stale_audio_output", "stale_generation_output"}


def _acceptance_runtime_factory(profile: str) -> Any:
    """Build the same production realtime factory used by the Task 13 server."""

    from src.runtime_app.container import build_production_runtime_factory

    config = ConfigLoader().load(profile)
    return build_production_runtime_factory(
        model_config=config["models"],
        audio_config=config.get("audio", {}),
    )


def _event_timestamp(event: Mapping[str, Any], clock: Callable[[], float]) -> float:
    for name in ("server_timestamp", "timestamp"):
        value = event.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float(clock())


def _benchmark_event_name(name: str) -> str | None:
    normalized = name.strip().lower()
    aliases = {
        "audio_frame_accepted": "audio_received",
        "transcript": "first_asr_partial",
        "token": "first_llm_token",
        "audio_chunk": "first_audio_chunk",
        "first_playable_audio": "first_audio_chunk",
    }
    if normalized in aliases:
        return aliases[normalized]
    return normalized if normalized in _KNOWN_CAPTURE_EVENTS else None


def _record_capture_event(
    timeline: BenchmarkTimeline,
    event: Mapping[str, Any],
    clock: Callable[[], float],
) -> str | None:
    name = _benchmark_event_name(str(event.get("event", event.get("name", ""))))
    if name is None:
        return None
    timeline.record(name, _event_timestamp(event, clock))
    return name


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _observe_task13_turn_end(
    runtime: Any,
    timeline: BenchmarkTimeline,
    clock: Callable[[], float],
) -> Callable[[], None]:
    """Observe the fused Task 13 turn boundary without changing runtime policy."""

    controller = getattr(runtime, "controller", None)
    original = getattr(controller, "handle_candidate", None)
    if controller is None or not callable(original):
        return lambda: None

    def observed(state: Any, candidate: Any) -> Any:
        if (
            getattr(candidate, "event", None) == "USER_TURN_END_CANDIDATE"
            and timeline.first("turn_end") is None
        ):
            timeline.record("turn_end", clock())
        return original(state, candidate)

    try:
        controller.handle_candidate = observed
    except (AttributeError, TypeError):
        return lambda: None

    def restore() -> None:
        controller.handle_candidate = original

    return restore


class RecordedAudioRealtimeDriver:
    """Feed recorded PCM through the Task 13 realtime session path."""

    def __init__(
        self,
        runtime_factory: AcceptanceRuntimeFactory | None = None,
        *,
        sleep: Callable[[float], Awaitable[None] | None] = asyncio.sleep,
        drain_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if drain_timeout <= 0:
            raise ValueError("drain_timeout must be positive")
        self.runtime_factory = runtime_factory or _acceptance_runtime_factory
        self.sleep = sleep
        self.drain_timeout = float(drain_timeout)
        self.clock = clock

    async def run(self, audio_path: str | Path, *, profile: str) -> AcceptanceEvidence:
        path = Path(audio_path)
        frames = self._read_frames(path)
        run_id = uuid.uuid4().hex
        factory = await _maybe_await(self.runtime_factory(profile))
        runtime = await _maybe_await(factory(run_id))
        timeline = BenchmarkTimeline()
        first_audio = asyncio.Event()
        restore_turn_observer = _observe_task13_turn_end(runtime, timeline, self.clock)

        async def collect_events() -> None:
            async for event in runtime.events():
                if not isinstance(event, Mapping):
                    continue
                if _record_capture_event(timeline, event, self.clock) == "first_audio_chunk":
                    first_audio.set()

        collector: asyncio.Task[None] | None = None
        runtime_started = False
        try:
            await _maybe_await(runtime.start())
            runtime_started = True
            collector = asyncio.create_task(collect_events(), name="recorded-audio-acceptance-events")
            timeline.record("audio_received", self.clock())
            frame_duration = PCM16_FRAME_SAMPLES / PCM16_SAMPLE_RATE
            capture_started = self.clock()
            for sequence, pcm in enumerate(frames):
                await runtime.accept_audio_frame(
                    RealtimeAudioFrame(
                        header=AudioFrameHeader(
                            sequence=sequence,
                            capture_timestamp=capture_started + sequence * frame_duration,
                        ),
                        pcm=pcm,
                    )
                )
                await asyncio.sleep(0)
                if sequence + 1 < len(frames):
                    await _maybe_await(self.sleep(frame_duration))
            flush = getattr(runtime, "flush", None)
            if callable(flush):
                await _maybe_await(flush())
            if timeline.first("turn_end") is None:
                timeline.record("turn_end", self.clock())

            audio_waiter = asyncio.create_task(first_audio.wait())
            done, pending = await asyncio.wait(
                {collector, audio_waiter},
                timeout=self.drain_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                if task is audio_waiter:
                    task.cancel()
            if audio_waiter in done:
                await audio_waiter
            elif audio_waiter.cancelled():
                await asyncio.gather(audio_waiter, return_exceptions=True)
        finally:
            restore_turn_observer()
            try:
                try:
                    if runtime_started:
                        await _maybe_await(runtime.close())
                finally:
                    if collector is not None:
                        try:
                            await asyncio.wait_for(collector, timeout=self.drain_timeout)
                        except asyncio.TimeoutError:
                            collector.cancel()
                            await asyncio.gather(collector, return_exceptions=True)
            finally:
                close_factory = getattr(factory, "close", None)
                if callable(close_factory):
                    await _maybe_await(close_factory())

        return AcceptanceEvidence(
            metrics=calculate_metrics(timeline),
            environment={
                "profile": profile,
                "audio": str(path.resolve()),
                "sample_rate": PCM16_SAMPLE_RATE,
                "channels": PCM16_MONO_CHANNELS,
            },
            provenance=AcceptanceProvenance.runner_capture(
                EvidenceSource.RECORDED_AUDIO_REALTIME,
                run_id=run_id,
            ),
            timeline=timeline,
        )

    @staticmethod
    def _read_frames(path: Path) -> tuple[bytes, ...]:
        try:
            with wave.open(str(path), "rb") as source:
                if source.getnchannels() != PCM16_MONO_CHANNELS:
                    raise ValueError("recorded audio must be mono")
                if source.getsampwidth() != 2:
                    raise ValueError("recorded audio must use PCM16 samples")
                if source.getframerate() != PCM16_SAMPLE_RATE:
                    raise ValueError(f"recorded audio must use {PCM16_SAMPLE_RATE} Hz")
                if source.getcomptype() != "NONE":
                    raise ValueError("recorded audio must be uncompressed PCM")
                pcm = source.readframes(source.getnframes())
        except (OSError, wave.Error) as exc:
            raise ValueError(f"unable to read recorded audio {path}: {exc}") from exc
        if not pcm:
            raise ValueError("recorded audio must contain at least one frame")
        chunks: list[bytes] = []
        for offset in range(0, len(pcm), PCM16_FRAME_BYTES):
            chunk = pcm[offset : offset + PCM16_FRAME_BYTES]
            chunks.append(chunk.ljust(PCM16_FRAME_BYTES, b"\x00"))
        return tuple(chunks)


class BrowserHeadsetAcceptanceDriver:
    """Run an external browser/headset capture and stamp local provenance."""

    def __init__(
        self,
        capture: BrowserCapture,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capture = capture
        self.clock = clock

    async def run(self, *, profile: str) -> AcceptanceEvidence:
        payload = await _maybe_await(self.capture(profile))
        if not isinstance(payload, Mapping):
            raise TypeError("browser/headset driver must return a mapping")
        timeline = BenchmarkTimeline()
        raw_timeline = payload.get("timeline", ())
        if not isinstance(raw_timeline, (list, tuple)):
            raise ValueError("browser/headset timeline must be a list")
        for event in raw_timeline:
            if not isinstance(event, Mapping):
                raise ValueError("browser/headset timeline entries must be objects")
            _record_capture_event(timeline, event, self.clock)
        supplied_metrics = payload.get("metrics", {})
        if not isinstance(supplied_metrics, Mapping):
            raise ValueError("browser/headset metrics must be an object")
        metrics = {name: float(value) for name, value in supplied_metrics.items()}
        metrics.update(calculate_metrics(timeline))
        environment = payload.get("environment", {})
        if not isinstance(environment, Mapping):
            raise ValueError("browser/headset environment must be an object")
        return AcceptanceEvidence(
            metrics=metrics,
            environment=dict(environment),
            provenance=AcceptanceProvenance.runner_capture(
                EvidenceSource.BROWSER_HEADSET,
                run_id=uuid.uuid4().hex,
            ),
            timeline=timeline,
        )


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
