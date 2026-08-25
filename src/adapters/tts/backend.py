"""Injectable streaming TTS backend adapter."""

import inspect
from typing import Any

from .base import BaseTTSAdapter


class StreamingTTSBackend(BaseTTSAdapter):
    """Adapt an injected provider without importing a provider SDK."""

    def __init__(self, provider: Any, model_path: str | None = None) -> None:
        if provider is None:
            raise ValueError("a TTS provider must be supplied")
        self.provider = provider
        self.model_path = model_path
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

    async def stream_audio(self, text: str, **options: Any) -> Any:
        """Forward streaming synthesis to the injected provider."""
        if self._interrupted:
            raise RuntimeError("TTS backend has been interrupted")
        method = self._provider_method("stream_audio", "synthesize_stream")
        result = self._call_supported(method, text, **options)
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
            result = self._call_supported(method)
            if inspect.isawaitable(result):
                await result

    async def cancel(self, request_id: str) -> None:
        """Forward request-scoped cancellation without changing interrupt()."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        self._interrupted = True
        method = getattr(self.provider, "cancel", None)
        if method is None:
            method = getattr(self.provider, "interrupt", None)
        if method is not None:
            result = self._call_supported(method, request_id)
            if inspect.isawaitable(result):
                await result

    def reset(self) -> Any:
        """Prepare the adapter for a new TTS session."""
        self._interrupted = False
        method = getattr(self.provider, "reset", None)
        if callable(method):
            return method()
        return None

    def _provider_method(self, *names: str):
        for name in names:
            method = getattr(self.provider, name, None)
            if callable(method):
                return method
        names_text = ", ".join(f"{name}()" for name in names)
        raise RuntimeError(f"TTS provider must expose one of: {names_text}.")

    @staticmethod
    def _call_supported(method: Any, *args: Any, **kwargs: Any) -> Any:
        """Pass realtime options to capable providers and preserve old ones."""
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


StreamingTTSAdapter = StreamingTTSBackend
