"""Whisper-shaped ASR provider wrapper with no hard dependency on an SDK.

The runtime object is injected so deployments can use a local Whisper SDK,
another compatible implementation, or a deterministic test double.  This
module owns provider result normalization and cancellation protection; the ASR
pipeline remains unaware of the provider.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any

from src.asr.stream import TranscriptChunk


class WhisperASRProvider:
    """Adapt a local Whisper-compatible runtime to transcript chunks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        options: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
        language: str | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("Whisper model_path must be supplied")
        if runtime is None:
            raise RuntimeError(
                "Whisper runtime is unavailable; install/configure a local runtime "
                "and inject it into WhisperASRProvider"
            )
        self.model_path = model_path
        self.device = device
        self.options = dict(options or {})
        if language is not None:
            self.options.setdefault("language", language)
        self.runtime = runtime
        self._cancelled = False

    async def stream_audio(self, audio_chunk: bytes) -> AsyncIterable[TranscriptChunk]:
        if self._cancelled:
            raise RuntimeError("Whisper ASR provider has been cancelled")
        method = self._runtime_method()
        result = method(audio_chunk, **self.options)
        if inspect.isawaitable(result):
            result = await result

        async def normalized() -> AsyncIterable[TranscriptChunk]:
            async for item in self._iterate(result):
                if self._cancelled:
                    return
                yield self._normalize(item)

        return normalized()

    async def cancel(self) -> None:
        self._cancelled = True
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        self._cancelled = False

    def _runtime_method(self):
        for name in ("stream_audio", "transcribe_stream", "transcribe"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        raise RuntimeError(
            "Whisper runtime must expose stream_audio(), transcribe_stream(), or transcribe()"
        )

    @staticmethod
    async def _iterate(result: Any) -> AsyncIterable[Any]:
        if result is None:
            return
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
            return
        if isinstance(result, (list, tuple)):
            for item in result:
                yield item
            return
        yield result

    @staticmethod
    def _normalize(result: Any) -> TranscriptChunk:
        if isinstance(result, TranscriptChunk):
            return result
        if isinstance(result, str):
            return TranscriptChunk("whisper-chunk", result, time.time(), False)
        if isinstance(result, Mapping):
            text = result.get("text", "")
            if not isinstance(text, str):
                raise TypeError("Whisper transcript text must be a string")
            return TranscriptChunk(
                str(result.get("chunk_id", "whisper-chunk")),
                text,
                float(result.get("timestamp", time.time())),
                bool(result.get("is_final", False)),
            )
        raise TypeError("Whisper result must be TranscriptChunk, text, mapping, or None")
