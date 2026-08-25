"""Rolling X2-Turn inference that publishes data-only turn candidates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import inspect
import math
import time
from types import MappingProxyType
from typing import Any

from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import (
    PCM16_FRAME_BYTES,
    PCM16_MONO_CHANNELS,
    PCM16_SAMPLE_RATE,
)


_LABEL_ALIASES = {
    "idle": "idle",
    "noidle": "noidle",
    "no_idle": "noidle",
    "nonidle": "noidle",
    "non_idle": "noidle",
    "speaking": "speaking",
    "speech": "speaking",
    "turn_end": "turn_end",
    "end_of_turn": "turn_end",
    "backchannel": "backchannel",
    "back_channel": "backchannel",
}

# A label-count mapping is a summary, not a time series. On equal counts we
# retain speech evidence before idle, backchannel, or turn-end evidence:
# speaking > noidle > idle > backchannel > turn_end.
AGGREGATE_LABEL_PRIORITY = (
    "speaking",
    "noidle",
    "idle",
    "backchannel",
    "turn_end",
)


@dataclass(frozen=True)
class TurnCandidate:
    """Immutable acoustic evidence for a later policy decision.

    ``metadata`` deliberately excludes transcript text: ASR remains the
    authoritative transcript source even when an X2 wrapper also returns one.
    """

    label: str
    confidence: float = 1.0
    capture_timestamp: float | None = None
    sequence: int | None = None
    revision_id: int | None = None
    inference_duration: float = 0.0
    rtf: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _normalize_label(self.label))
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        for name in ("capture_timestamp", "inference_duration", "rtf"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(value, name))
        if self.sequence is not None and (type(self.sequence) is not int or self.sequence < 0):
            raise ValueError("sequence must be a nonnegative integer or None")
        if self.revision_id is not None and (
            type(self.revision_id) is not int or self.revision_id < 0
        ):
            raise ValueError("revision_id must be a nonnegative integer or None")
        if self.inference_duration < 0 or self.rtf < 0:
            raise ValueError("inference duration and RTF must not be negative")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-safe representation without mutable internals."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "capture_timestamp": self.capture_timestamp,
            "sequence": self.sequence,
            "revision_id": self.revision_id,
            "inference_duration": self.inference_duration,
            "rtf": self.rtf,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _NormalizedTurnLabel:
    """One normalized label before provider-specific identity enrichment."""

    label: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capture_timestamp: float | None = None
    sequence: int | None = None
    revision_id: int | None = None


def normalize_turn_candidates(
    result: Any,
    *,
    capture_timestamp: float | None = None,
    sequence: int | None = None,
    revision_id: int | None = None,
    inference_duration: float = 0.0,
    rtf: float = 0.0,
    next_revision_id: Callable[[], int] | None = None,
) -> tuple[TurnCandidate, ...]:
    """Normalize supported X2 wrapper output into data-only candidates.

    Ordered frame lists retain their order. Label-count mappings intentionally
    become exactly one aggregate candidate because their counts have no
    temporal ordering information.
    """
    normalized = _normalize_turn_result(result)
    candidates = []
    for item in normalized:
        candidate_revision = (
            next_revision_id() if next_revision_id is not None else revision_id
        )
        candidates.append(
            TurnCandidate(
                item.label,
                item.confidence,
                capture_timestamp=(
                    capture_timestamp
                    if capture_timestamp is not None
                    else item.capture_timestamp
                ),
                sequence=sequence if sequence is not None else item.sequence,
                revision_id=(
                    candidate_revision
                    if candidate_revision is not None
                    else item.revision_id
                ),
                inference_duration=inference_duration,
                rtf=rtf,
                metadata=item.metadata,
            )
        )
    return tuple(candidates)


class X2TurnRollingProvider:
    """Run an injected X2 inference runtime on a short PCM16 rolling window."""

    def __init__(
        self,
        model_path: str,
        *,
        runtime: Any | None = None,
        cadence_ms: int = 160,
        context_seconds: float = 2.0,
    ) -> None:
        if not model_path:
            raise ValueError("X2-Turn model_path must be supplied")
        if not 100 <= cadence_ms <= 200:
            raise ValueError("cadence_ms must be between 100 and 200")
        if not 1.0 <= context_seconds <= 3.0:
            raise ValueError("context_seconds must be between 1 and 3")

        self.model_path = model_path
        self.runtime = runtime
        self.cadence_ms = cadence_ms
        self.context_seconds = context_seconds
        self._max_context_bytes = int(context_seconds * PCM16_SAMPLE_RATE * 2)
        self._cadence_bytes = int(cadence_ms * PCM16_SAMPLE_RATE * 2 / 1000)
        self._audio = bytearray()
        self._bytes_since_decode = 0
        self._latest_frame: RealtimeAudioFrame | None = None
        self._cancelled = False
        self._generation = 0
        self._publication_id = 0
        self._decode_lock = asyncio.Lock()
        self.decode_durations: list[float] = []
        self.total_decode_seconds = 0.0
        self.source_audio_seconds = 0.0
        self.last_decode_duration = 0.0
        self.last_rtf = 0.0

    @property
    def context_bytes(self) -> bytes:
        """Return a copy of the currently retained PCM16 context."""
        return bytes(self._audio)

    @property
    def publication_id(self) -> int:
        """Return the latest monotonic candidate publication identity."""
        return self._publication_id

    @property
    def end_to_end_decode_rtf(self) -> float:
        """Return aggregate inference time divided by all accepted audio time."""
        if not self.source_audio_seconds:
            return 0.0
        return self.total_decode_seconds / self.source_audio_seconds

    async def push_pcm(self, pcm: bytes | RealtimeAudioFrame) -> tuple[TurnCandidate, ...]:
        """Accept one exact V1 PCM frame and infer only when cadence is due."""
        frame_pcm, frame = _validated_pcm(pcm)
        async with self._decode_lock:
            self._require_active()
            self._append(frame_pcm, frame)
            if self._bytes_since_decode < self._cadence_bytes:
                return ()
            return await self._decode()

    async def finalize_turn(self) -> tuple[TurnCandidate, ...]:
        """Force an inference on the remaining current turn audio."""
        async with self._decode_lock:
            self._require_active()
            if not self._audio:
                return ()
            return await self._decode()

    async def cancel(self) -> None:
        """Invalidate in-flight inference and discard the current turn window."""
        self._cancelled = True
        self._generation += 1
        async with self._decode_lock:
            self._reset_turn_state()
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Start a new turn without reusing previously published identities."""
        self._cancelled = False
        self._generation += 1
        self._reset_turn_state()

    async def _decode(self) -> tuple[TurnCandidate, ...]:
        generation = self._generation
        audio = bytes(self._audio)
        latest_frame = self._latest_frame
        started = time.perf_counter()
        result = await self._invoke_runtime(audio)
        duration = time.perf_counter() - started
        self.decode_durations.append(duration)
        self.total_decode_seconds += duration
        self.last_decode_duration = duration
        self.last_rtf = duration / (len(audio) / (PCM16_SAMPLE_RATE * 2)) if audio else 0.0
        if self._cancelled or generation != self._generation:
            raise RuntimeError("X2-Turn provider has been cancelled")

        candidates = await asyncio.to_thread(
            normalize_turn_candidates,
            result,
            capture_timestamp=(
                latest_frame.capture_timestamp if latest_frame is not None else None
            ),
            sequence=latest_frame.sequence if latest_frame is not None else None,
            inference_duration=duration,
            rtf=self.last_rtf,
            next_revision_id=self._next_publication_id,
        )
        self._bytes_since_decode = 0
        return candidates

    async def _invoke_runtime(self, audio: bytes) -> Any:
        method = self._runtime_method()
        if inspect.iscoroutinefunction(method):
            result = await method(audio)
        else:
            result = await asyncio.to_thread(method, audio)
            if inspect.isawaitable(result):
                result = await result
        if hasattr(result, "__aiter__"):
            items = []
            async for item in result:
                items.append(item)
            result = items
        return result

    def _runtime_method(self):
        if self.runtime is None:
            raise RuntimeError("X2-Turn runtime is unavailable; inject a runtime")
        for name in ("infer", "infer_turn", "process_audio", "push_audio"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        if callable(self.runtime):
            return self.runtime
        raise RuntimeError("X2-Turn runtime must expose infer() or process_audio()")

    def _append(self, pcm: bytes, frame: RealtimeAudioFrame | None) -> None:
        self._audio.extend(pcm)
        self._bytes_since_decode += len(pcm)
        self.source_audio_seconds += len(pcm) / (PCM16_SAMPLE_RATE * 2)
        if frame is not None:
            self._latest_frame = frame
        if len(self._audio) > self._max_context_bytes:
            del self._audio[: len(self._audio) - self._max_context_bytes]

    def _require_active(self) -> None:
        if self._cancelled:
            raise RuntimeError("X2-Turn provider has been cancelled")

    def _reset_turn_state(self) -> None:
        self._audio.clear()
        self._bytes_since_decode = 0
        self._latest_frame = None

    def _next_publication_id(self) -> int:
        self._publication_id += 1
        return self._publication_id


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        raise TypeError("turn label must be a string")
    normalized = "_".join(label.strip().lower().replace("-", " ").split())
    candidate = _LABEL_ALIASES.get(normalized)
    if candidate is None:
        raise ValueError(f"unknown X2-Turn label: {label!r}")
    return candidate


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validated_pcm(value: bytes | RealtimeAudioFrame) -> tuple[bytes, RealtimeAudioFrame | None]:
    if isinstance(value, RealtimeAudioFrame):
        if value.header.sample_rate != PCM16_SAMPLE_RATE:
            raise ValueError("X2-Turn requires 16kHz PCM")
        if value.header.channels != PCM16_MONO_CHANNELS:
            raise ValueError("X2-Turn requires mono PCM")
        pcm = value.pcm
        frame: RealtimeAudioFrame | None = value
    elif isinstance(value, bytes):
        pcm = value
        frame = None
    else:
        raise TypeError("PCM input must be bytes or RealtimeAudioFrame")
    if len(pcm) != PCM16_FRAME_BYTES:
        raise ValueError(f"PCM input must be exactly {PCM16_FRAME_BYTES} bytes")
    if len(pcm) % 2:
        raise ValueError("PCM input must contain PCM16 samples")
    return pcm, frame


def _normalize_turn_result(result: Any) -> tuple[_NormalizedTurnLabel, ...]:
    """Normalize official/test X2 wrapper result shapes without transcript use."""
    if result is None:
        return ()
    if isinstance(result, tuple) and len(result) == 2:
        # The official wrapper can return ``(transcript, turn_frames)``.
        return _normalize_turn_result(result[1])
    if isinstance(result, str):
        return (_NormalizedTurnLabel(_normalize_label(result), 1.0),)
    if isinstance(result, Mapping):
        if _label_value(result) is not None:
            return (_normalize_turn_item(result),)
        for name in ("turn_frames", "turn_labels", "labels", "turns"):
            if name in result:
                return _normalize_turn_collection(result[name])
        return ()
    for name in ("turn_frames", "turn_labels", "labels", "turns"):
        value = getattr(result, name, None)
        if value is not None:
            return _normalize_turn_collection(value)
    if _label_value(result) is not None:
        return (_normalize_turn_item(result),)
    if isinstance(result, Iterable) and not isinstance(result, (bytes, bytearray)):
        normalized = []
        for item in result:
            normalized.extend(_normalize_turn_result(item))
        return tuple(normalized)
    raise TypeError("X2-Turn result must expose turn labels")


def _normalize_turn_collection(value: Any) -> tuple[_NormalizedTurnLabel, ...]:
    if isinstance(value, Mapping) and _label_value(value) is None:
        counts: dict[str, int] = {}
        for label, count in value.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TypeError("turn label counts must be nonnegative integers")
            if count:
                normalized_label = _normalize_label(str(label))
                counts[normalized_label] = counts.get(normalized_label, 0) + count
        if not counts:
            return ()
        dominant = max(
            counts,
            key=lambda label: (counts[label], -AGGREGATE_LABEL_PRIORITY.index(label)),
        )
        total = sum(counts.values())
        return (
            _NormalizedTurnLabel(
                dominant,
                counts[dominant] / total,
                {"aggregated": True, "counts": dict(counts)},
            ),
        )
    return _normalize_turn_result(value)


def _normalize_turn_item(item: Any) -> _NormalizedTurnLabel:
    if isinstance(item, Mapping):
        values = item
        label = _label_value(values)
        confidence = values.get("confidence", values.get("score", values.get("probability", 1.0)))
        count = values.get("count")
        capture_timestamp = values.get("capture_timestamp", values.get("timestamp"))
        sequence = values.get("sequence")
        revision_id = values.get("revision_id")
    else:
        label = _label_value(item)
        confidence = getattr(item, "confidence", getattr(item, "score", 1.0))
        count = getattr(item, "count", None)
        capture_timestamp = getattr(item, "capture_timestamp", getattr(item, "timestamp", None))
        sequence = getattr(item, "sequence", None)
        revision_id = getattr(item, "revision_id", None)
    if label is None:
        raise ValueError("X2-Turn label item must include a label")
    metadata = {"count": count} if count is not None else {}
    return _NormalizedTurnLabel(
        _normalize_label(label),
        _validated_confidence(confidence),
        metadata,
        capture_timestamp,
        sequence,
        revision_id,
    )


def _label_value(item: Any) -> str | None:
    if isinstance(item, Mapping):
        for name in ("label", "state", "event", "type"):
            value = item.get(name)
            if isinstance(value, str):
                return value
        return None
    for name in ("label", "state", "event", "type"):
        value = getattr(item, name, None)
        if isinstance(value, str):
            return value
    return None


def _validated_confidence(value: Any) -> float:
    confidence = _finite_number(value, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence
