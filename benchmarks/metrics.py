"""Deterministic latency calculations and acceptance gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .timeline import BenchmarkTimeline


METRIC_EVENTS: Mapping[str, tuple[str, str]] = {
    "first_transcript_latency_ms": ("audio_received", "first_asr_partial"),
    "duck_latency_ms": ("user_speech_start", "assistant_ducked"),
    "first_token_latency_ms": ("turn_end", "first_llm_token"),
    "first_audio_latency_ms": ("first_llm_token", "first_audio_chunk"),
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
    explicit_real_run: bool = True,
    required_measurements_present: bool = True,
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
    if label == "hardware-e2e":
        if not explicit_real_run:
            failures.append("hardware_e2e_requires_explicit_real_run")
        if not required_measurements_present:
            failures.append("hardware_e2e_missing_required_measurements")
    return AcceptanceResult(
        label=label,
        passed=not failures,
        failures=tuple(failures),
        metrics=normalized,
    )
