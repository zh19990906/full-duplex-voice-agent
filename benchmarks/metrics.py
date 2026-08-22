"""Deterministic latency calculations over :mod:`benchmarks.timeline`."""

from __future__ import annotations

from typing import Mapping

from .timeline import BenchmarkTimeline


METRIC_EVENTS: Mapping[str, tuple[str, str]] = {
    "first_transcript_latency_ms": ("audio_received", "first_asr_partial"),
    "first_token_latency_ms": ("turn_end", "first_llm_token"),
    "first_audio_latency_ms": ("first_llm_token", "first_audio_chunk"),
    "interrupt_latency_ms": ("user_interrupt", "tts_stopped"),
    "cancellation_latency_ms": ("cancel_requested", "generation_cancelled"),
    "cancel_latency_ms": ("cancel_requested", "generation_cancelled"),
    "first_translated_audio_latency_ms": (
        "translation_started",
        "first_translated_audio",
    ),
    "resume_transition_latency_ms": ("resume_requested", "task_resumed"),
    "generation_continuation_latency_ms": (
        "backchannel_received",
        "generation_continued",
    ),
    "startup_latency_ms": ("startup_started", "runtime_ready"),
    "audio_buffer_wait_ms": ("audio_frame_enqueued", "audio_frame_routed"),
    "token_queue_wait_ms": ("token_enqueued", "token_synthesized"),
}


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
