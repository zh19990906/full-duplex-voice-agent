"""Rolling, revision-aware adapter for local faster-whisper ASR."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Iterable, Mapping
from typing import Any

from src.asr.stream import TranscriptChunk
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import (
    PCM16_FRAME_BYTES,
    PCM16_MONO_CHANNELS,
    PCM16_SAMPLE_RATE,
)
from src.realtime.stable_prefix import StablePrefixCommitter


class FasterWhisperStreamingProvider:
    """Decode a bounded PCM16 window and publish stable transcript deltas."""

    def __init__(
        self,
        model_path: str,
        *,
        runtime: Any | None = None,
        load_model: bool = False,
        cadence_ms: int = 300,
        context_seconds: float = 5.0,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("faster-whisper model_path must be supplied")
        if not 200 <= cadence_ms <= 400:
            raise ValueError("cadence_ms must be between 200 and 400")
        if not 0 < context_seconds <= 10:
            raise ValueError("context_seconds must be greater than zero and at most ten")

        self.model_path = model_path
        self.cadence_ms = cadence_ms
        self.context_seconds = context_seconds
        self.device = device
        self.compute_type = compute_type
        self.options = dict(options or {})
        if language is not None:
            self.options.setdefault("language", language)
        self.runtime = runtime
        if self.runtime is None and load_model:
            self.runtime = _load_faster_whisper_runtime(
                model_path, device=device, compute_type=compute_type
            )

        self._max_context_bytes = int(context_seconds * PCM16_SAMPLE_RATE * 2)
        self._cadence_bytes = int(cadence_ms * PCM16_SAMPLE_RATE * 2 / 1000)
        self._audio = bytearray()
        self._bytes_since_decode = 0
        self._committer = StablePrefixCommitter()
        self._revision_id = 0
        self._cancelled = False
        self._turn_generation = 0
        self._decode_lock = asyncio.Lock()
        self.decode_durations: list[float] = []
        self.last_decode_duration = 0.0
        self.last_rtf = 0.0

    @property
    def context_bytes(self) -> bytes:
        """Return a copy of the bounded rolling PCM context for observability."""
        return bytes(self._audio)

    async def push_pcm(self, pcm: bytes | RealtimeAudioFrame) -> TranscriptChunk | None:
        """Accept one ordered V1 PCM frame and decode when cadence is due."""
        frame_pcm = _validated_pcm(pcm)
        async with self._decode_lock:
            self._require_active()
            self._append(frame_pcm)
            if self._bytes_since_decode < self._cadence_bytes:
                return None
            return await self._decode(final=False)

    async def finalize_turn(self) -> TranscriptChunk:
        """Force one final decode, emit its remaining delta, and reset the turn."""
        async with self._decode_lock:
            self._require_active()
            if self._audio:
                chunk = await self._decode(final=True)
            else:
                self._revision_id += 1
                chunk = TranscriptChunk(
                    f"faster-whisper-{self._revision_id}",
                    "",
                    time.time(),
                    True,
                    self._revision_id,
                    "",
                )
            self._reset_turn_state()
            return chunk

    async def cancel(self) -> None:
        """Invalidate active decode results and discard this turn's state."""
        self._cancelled = True
        self._turn_generation += 1
        async with self._decode_lock:
            self._reset_turn_state()
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Enable a fresh turn; a pre-reset decode cannot publish afterward."""
        self._cancelled = False
        self._turn_generation += 1
        self._reset_turn_state()

    async def _decode(self, *, final: bool) -> TranscriptChunk:
        generation = self._turn_generation
        audio = bytes(self._audio)
        started = time.perf_counter()
        result = await self._invoke_runtime(audio)
        duration = time.perf_counter() - started
        self.decode_durations.append(duration)
        self.last_decode_duration = duration
        self.last_rtf = duration / (len(audio) / (PCM16_SAMPLE_RATE * 2)) if audio else 0.0
        if self._cancelled or generation != self._turn_generation:
            raise RuntimeError("faster-whisper ASR provider has been cancelled")

        hypothesis = _normalize_text(result)
        if final:
            stable_delta = self._committer.finalize(hypothesis)
            unstable_text = ""
        else:
            stable_delta = self._committer.update(hypothesis)
            unstable_text = self._committer.unstable_text
        self._revision_id += 1
        self._bytes_since_decode = 0
        return TranscriptChunk(
            f"faster-whisper-{self._revision_id}",
            stable_delta,
            time.time(),
            final,
            self._revision_id,
            unstable_text,
        )

    async def _invoke_runtime(self, audio: bytes) -> Any:
        method = self._runtime_method()
        if inspect.iscoroutinefunction(method):
            return await method(audio, **self.options)
        result = await asyncio.to_thread(method, audio, **self.options)
        if inspect.isawaitable(result):
            return await result
        return result

    def _runtime_method(self):
        if self.runtime is None:
            raise RuntimeError(
                "faster-whisper runtime is unavailable; inject one or pass load_model=True"
            )
        for name in ("transcribe", "transcribe_stream", "stream_audio"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        if callable(self.runtime):
            return self.runtime
        raise RuntimeError("faster-whisper runtime must expose transcribe()")

    def _append(self, pcm: bytes) -> None:
        self._audio.extend(pcm)
        if len(self._audio) > self._max_context_bytes:
            del self._audio[: len(self._audio) - self._max_context_bytes]
        self._bytes_since_decode += len(pcm)

    def _require_active(self) -> None:
        if self._cancelled:
            raise RuntimeError("faster-whisper ASR provider has been cancelled")

    def _reset_turn_state(self) -> None:
        self._audio.clear()
        self._bytes_since_decode = 0
        self._committer.reset()
        self._revision_id = 0


def _validated_pcm(value: bytes | RealtimeAudioFrame) -> bytes:
    if isinstance(value, RealtimeAudioFrame):
        if value.header.sample_rate != PCM16_SAMPLE_RATE:
            raise ValueError("faster-whisper requires 16kHz PCM")
        if value.header.channels != PCM16_MONO_CHANNELS:
            raise ValueError("faster-whisper requires mono PCM")
        pcm = value.pcm
    elif isinstance(value, bytes):
        pcm = value
    else:
        raise TypeError("PCM input must be bytes or RealtimeAudioFrame")
    if len(pcm) != PCM16_FRAME_BYTES:
        raise ValueError(f"PCM input must be exactly {PCM16_FRAME_BYTES} bytes")
    if len(pcm) % 2:
        raise ValueError("PCM input must contain PCM16 samples")
    return pcm


def _normalize_text(result: Any) -> str:
    """Extract text from faster-whisper's tuple, segment, or test-double shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        text = result.get("text", "")
        if not isinstance(text, str):
            raise TypeError("faster-whisper mapping text must be a string")
        return text
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, tuple) and len(result) == 2:
        return _normalize_text(result[0])
    if isinstance(result, Iterable):
        return "".join(_normalize_text(item) for item in result)
    raise TypeError("faster-whisper result must expose transcript text")


def _load_faster_whisper_runtime(
    model_path: str, *, device: str, compute_type: str
) -> Any:
    """Load the optional SDK only after an explicit model-loading request."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - deployment-only dependency
        raise RuntimeError("install faster-whisper to load a local ASR model") from error

    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    return _FasterWhisperModelRuntime(model)


class _FasterWhisperModelRuntime:
    """Convert V1 PCM16 bytes into the float waveform expected by the SDK."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(self, pcm: bytes, **options: Any) -> Any:
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("numpy is required for faster-whisper PCM conversion") from error
        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        return self._model.transcribe(audio, **options)
