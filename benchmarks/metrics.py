"""Deterministic latency calculations and acceptance gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .timeline import BenchmarkTimeline


METRIC_EVENTS: Mapping[str, tuple[str, str]] = {
    "first_transcript_latency_ms": ("audio_received", "first_asr_partial"),
    "duck_latency_ms": ("user_speech_start", "assistant_ducked"),
    "first_token_latency_ms": ("turn_end", "first_llm_token"),
    "first_audio_latency_ms": ("turn_end", "first_audio_chunk"),
    "interrupt_latency_ms": ("user_interrupt", "tts_stopped"),
    "cancellation_latency_ms": ("cancel_requested", "generation_cancelled"),
    "cancel_latency_ms": ("cancel_requested", "generation_cancelled"),
    "first_translated_audio_latency_ms": (
        "translation_started",
        "first_translated_audio",
    ),
    "backchannel_restore_latency_ms": ("backchannel_finished", "assistant_restored"),
    "resume_transition_latency_ms": ("resume_requested", "task_resumed"),
    "generation_continuation_latency_ms": (
        "backchannel_received",
        "generation_continued",
    ),
    "startup_latency_ms": ("startup_started", "runtime_ready"),
    "audio_buffer_wait_ms": ("audio_frame_enqueued", "audio_frame_routed"),
    "token_queue_wait_ms": ("token_enqueued", "token_synthesized"),
}

ACCEPTANCE_LABELS = frozenset({"unit", "simulated", "model-integration", "hardware-e2e"})
ACCEPTANCE_THRESHOLDS: Mapping[str, float] = {
    "duck_latency_ms": 100.0,
    "interrupt_latency_ms": 250.0,
    "backchannel_restore_latency_ms": 300.0,
    "first_token_latency_ms": 800.0,
    "first_audio_latency_ms": 1500.0,
    "first_translated_audio_latency_ms": 2000.0,
    "stale_output_count": 0.0,
    "resume_phrase_error_count": 1.0,
}


class EvidenceSource(str, Enum):
    """Capture paths that may produce high-tier acceptance evidence."""

    RECORDED_AUDIO_REALTIME = "recorded-audio-realtime"
    BROWSER_HEADSET = "browser-headset"


@dataclass(frozen=True)
class AcceptanceProvenance:
    """Runner-issued identity for one in-process acceptance capture."""

    source: EvidenceSource
    run_id: str
    producer: str

    RUNNER_PRODUCER = "realtime-acceptance-runner"

    @classmethod
    def runner_capture(
        cls,
        source: EvidenceSource,
        *,
        run_id: str,
    ) -> "AcceptanceProvenance":
        if not isinstance(source, EvidenceSource):
            raise TypeError("source must be an EvidenceSource")
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        return cls(source=source, run_id=run_id, producer=cls.RUNNER_PRODUCER)

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source.value,
            "run_id": self.run_id,
            "producer": self.producer,
        }


class _AcceptanceEvidenceCapability:
    """Opaque in-process proof that a runner completed a capture."""

    __slots__ = ("source", "run_id", "_seal")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("acceptance evidence capabilities are runner-issued")


def _capability_authority():
    seal = object()

    def issue(source: EvidenceSource, run_id: str) -> _AcceptanceEvidenceCapability:
        capability = object.__new__(_AcceptanceEvidenceCapability)
        capability.source = source
        capability.run_id = run_id
        capability._seal = seal
        return capability

    def matches(
        capability: object | None,
        provenance: AcceptanceProvenance | None,
        source: EvidenceSource,
    ) -> bool:
        return (
            isinstance(capability, _AcceptanceEvidenceCapability)
            and capability._seal is seal
            and capability.source is source
            and provenance is not None
            and provenance.source is source
            and provenance.run_id == capability.run_id
            and provenance.producer == AcceptanceProvenance.RUNNER_PRODUCER
        )

    return issue, matches


_issue_completed_capture, _completed_capture_matches = _capability_authority()


@dataclass(frozen=True)
class AcceptanceResult:
    label: str
    passed: bool
    failures: tuple[str, ...]
    metrics: dict[str, float]


def latency_ms(timeline: BenchmarkTimeline, start: str, end: str) -> float | None:
    """Return the first matching event interval in milliseconds."""

    start_entry = timeline.first(start)
    end_entry = timeline.first(end)
    if start_entry is None or end_entry is None:
        return None
    return round((end_entry.timestamp - start_entry.timestamp) * 1000.0, 6)


def calculate_metrics(timeline: BenchmarkTimeline) -> dict[str, float]:
    """Calculate all available metrics, omitting incomplete intervals."""

    metrics: dict[str, float] = {}
    for metric_name, (start, end) in METRIC_EVENTS.items():
        value = latency_ms(timeline, start, end)
        if value is not None:
            metrics[metric_name] = value
    metrics["stale_output_count"] = float(
        timeline.count("stale_audio_output") + timeline.count("stale_generation_output")
    )
    return metrics


def evaluate_acceptance(
    metrics: Mapping[str, float],
    *,
    label: str,
    provenance: AcceptanceProvenance | None = None,
    _evidence_capability: object | None = None,
) -> AcceptanceResult:
    """Evaluate V1 hard gates for an explicitly labeled evidence tier."""

    if label not in ACCEPTANCE_LABELS:
        raise ValueError(f"label must be one of {sorted(ACCEPTANCE_LABELS)}")
    normalized = {name: float(value) for name, value in dict(metrics).items()}
    failures: list[str] = []
    for metric_name, limit in ACCEPTANCE_THRESHOLDS.items():
        if metric_name not in normalized:
            failures.append(metric_name)
            continue
        value = normalized[metric_name]
        if metric_name == "stale_output_count":
            if value != limit:
                failures.append(metric_name)
            continue
        if value > limit:
            failures.append(metric_name)
    if label == "model-integration" and not _completed_capture_matches(
        _evidence_capability,
        provenance,
        EvidenceSource.RECORDED_AUDIO_REALTIME,
    ):
        failures.append("model_integration_requires_recorded_audio_runner")
    if label == "hardware-e2e" and not _completed_capture_matches(
        _evidence_capability,
        provenance,
        EvidenceSource.BROWSER_HEADSET,
    ):
        failures.append("hardware_e2e_requires_browser_headset_runner")
    return AcceptanceResult(
        label=label,
        passed=not failures,
        failures=tuple(failures),
        metrics=normalized,
    )
