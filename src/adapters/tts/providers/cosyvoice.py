"""CosyVoice-shaped streaming TTS provider wrapper without SDK dependency."""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any

from src.tts_runtime.stream import AudioChunk


class CosyVoiceTTSProvider:
    """Adapt an injected local CosyVoice-compatible runtime to ``AudioChunk``."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        options: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
        speaker: str | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("CosyVoice model_path must be supplied")
        if runtime is None:
            raise RuntimeError(
                "CosyVoice runtime is unavailable; configure a local runtime "
                "and inject it into CosyVoiceTTSProvider"
            )
        self.model_path = model_path
        self.device = device
        self.options = dict(options or {})
        if speaker is not None:
            self.options.setdefault("speaker", speaker)
        self.runtime = runtime
        self._interrupted = False

    async def synthesize(self, text: str) -> None:
        """Run optional provider preparation without requiring it."""

        if self._interrupted:
            raise RuntimeError("CosyVoice provider has been interrupted")
        method = getattr(self.runtime, "synthesize", None)
        if callable(method):
            result = method(text, **self.options)
            if inspect.isawaitable(result):
                await result

    async def stream_audio(self, text: str) -> AsyncIterable[AudioChunk]:
        if self._interrupted:
            raise RuntimeError("CosyVoice provider has been interrupted")
        method = self._runtime_method()
        result = method(text, **self.options)
        if inspect.isawaitable(result):
            result = await result

        async def normalized() -> AsyncIterable[AudioChunk]:
            async for item in self._iterate(result):
                if self._interrupted:
                    return
                yield self._normalize(item)

        return normalized()

    async def interrupt(self) -> None:
        self._interrupted = True
        method = getattr(self.runtime, "interrupt", None)
        if method is None:
            method = getattr(self.runtime, "cancel", None)
        if callable(method):
            result = method()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        self._interrupted = False

    def _runtime_method(self):
        for name in ("stream_audio", "synthesize_stream"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        raise RuntimeError("CosyVoice runtime must expose stream_audio() or synthesize_stream()")

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
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
            for item in result:
                yield item
            return
        yield result

    @staticmethod
    def _normalize(result: Any) -> AudioChunk:
        if isinstance(result, AudioChunk):
            return result
        if isinstance(result, bytes):
            return AudioChunk("cosyvoice-audio", result, time.time(), False)
        if isinstance(result, Mapping):
            audio_data = result.get("audio_data", result.get("data", b""))
            if not isinstance(audio_data, bytes):
                raise TypeError("CosyVoice audio data must be bytes")
            return AudioChunk(
                str(result.get("chunk_id", "cosyvoice-audio")),
                audio_data,
                float(result.get("timestamp", time.time())),
                bool(result.get("is_final", False)),
            )
        raise TypeError("CosyVoice result must be AudioChunk, bytes, or mapping")
