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

    async def stream_audio(self, text: str, **options: Any) -> AsyncIterable[AudioChunk]:
        """Stream audio while allowing realtime call identity to reach runtime."""
        if self._interrupted:
            raise RuntimeError("CosyVoice provider has been interrupted")
        method = self._runtime_method()
        runtime_options = dict(self.options)
        runtime_options.update(options)
        result = self._call_supported(method, text, **runtime_options)
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
            result = self._call_supported(method)
            if inspect.isawaitable(result):
                await result

    async def cancel(self, request_id: str) -> None:
        """Forward a request-scoped cancel when the wrapped runtime supports it."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        self._interrupted = True
        method = getattr(self.runtime, "cancel", None)
        if method is None:
            method = getattr(self.runtime, "interrupt", None)
        if callable(method):
            result = self._call_supported(method, request_id)
            if inspect.isawaitable(result):
                await result

    def reset(self) -> Any:
        self._interrupted = False
        method = getattr(self.runtime, "reset", None)
        if callable(method):
            return method()
        return None

    def _runtime_method(self):
        for name in ("stream_audio", "synthesize_stream"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        raise RuntimeError("CosyVoice runtime must expose stream_audio() or synthesize_stream()")

    @staticmethod
    def _call_supported(method: Any, *args: Any, **kwargs: Any) -> Any:
        """Retain compatibility with runtimes that do not know realtime keys."""

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return method(*args, **kwargs)
        positional_count = sum(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        return method(
            *args[:positional_count],
            **{key: value for key, value in kwargs.items() if key in signature.parameters},
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
