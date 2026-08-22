"""Injectable streaming TTS backend adapter."""

import inspect
from typing import Any

from .base import BaseTTSAdapter


class StreamingTTSBackend(BaseTTSAdapter):
    """Adapt an injected provider without importing a provider SDK."""

    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("a TTS provider must be supplied")
        self.provider = provider
        self._interrupted = False

    async def synthesize(self, text: str) -> Any:
        """Forward synthesis preparation to the injected provider."""
        if self._interrupted:
            raise RuntimeError("TTS backend has been interrupted")
        method = getattr(self.provider, "synthesize", None)
        if method is None:
            return None
        result = method(text)
        if inspect.isawaitable(result):
            return await result
        return result

    async def stream_audio(self, text: str) -> Any:
        """Forward streaming synthesis to the injected provider."""
        if self._interrupted:
            raise RuntimeError("TTS backend has been interrupted")
        method = self._provider_method("stream_audio", "synthesize_stream")
        result = method(text)
        if inspect.isawaitable(result):
            return await result
        return result

    async def interrupt(self) -> None:
        """Interrupt provider output and reject new text until reset."""
        self._interrupted = True
        method = getattr(self.provider, "interrupt", None)
        if method is None:
            method = getattr(self.provider, "cancel", None)
        if method is not None:
            result = method()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Prepare the adapter for a new TTS session."""
        self._interrupted = False

    def _provider_method(self, *names: str):
        for name in names:
            method = getattr(self.provider, name, None)
            if callable(method):
                return method
        names_text = ", ".join(f"{name}()" for name in names)
        raise RuntimeError(f"TTS provider must expose one of: {names_text}.")


StreamingTTSAdapter = StreamingTTSBackend
