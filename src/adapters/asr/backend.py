"""Injectable streaming ASR backend adapter."""

import inspect
from typing import Any

from .base import BaseASRAdapter


class StreamingASRBackend(BaseASRAdapter):
    """Adapt a provider-shaped object without importing any model SDK.

    The provider is deliberately injected. It may expose ``stream_audio``,
    ``process_audio``, or ``transcribe`` and may return normal or asynchronous
    transcript results. Provider-specific objects never leave this adapter.
    """

    def __init__(self, provider: Any, model_path: str | None = None) -> None:
        if provider is None:
            raise ValueError("an ASR provider must be supplied")
        self.provider = provider
        self.model_path = model_path
        self._cancelled = False

    async def stream_audio(self, audio_chunk: bytes) -> Any:
        """Forward audio to the injected provider's streaming entry point."""
        if self._cancelled:
            raise RuntimeError("ASR backend has been cancelled")
        method = self._provider_method()
        result = method(audio_chunk)
        if inspect.isawaitable(result):
            return await result
        return result

    async def cancel(self) -> None:
        """Cancel provider work and reject subsequent audio until reset."""
        self._cancelled = True
        cancel = getattr(self.provider, "cancel", None)
        if cancel is not None:
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Prepare the adapter for a new runtime session."""
        self._cancelled = False

    def _provider_method(self):
        for name in ("stream_audio", "process_audio", "transcribe"):
            method = getattr(self.provider, name, None)
            if callable(method):
                return method
        raise RuntimeError(
            "ASR provider must expose stream_audio(), process_audio(), or transcribe()."
        )


StreamingASRAdapter = StreamingASRBackend
