"""Injectable streaming LLM backend adapter."""

import inspect
from typing import Any

from .base import BaseLLMAdapter


class StreamingLLMBackend(BaseLLMAdapter):
    """Adapt an injected provider without importing a provider SDK.

    The provider may return normal values or an asynchronous token iterable.
    Provider-specific objects remain behind this adapter boundary.
    """

    def __init__(self, provider: Any, model_path: str | None = None) -> None:
        if provider is None:
            raise ValueError("an LLM provider must be supplied")
        self.provider = provider
        self.model_path = model_path
        self._cancelled = False

    async def generate(self, prompt: str) -> str:
        """Forward complete generation to the injected provider."""
        if self._cancelled:
            raise RuntimeError("LLM backend has been cancelled")
        method = self._provider_method("generate", "complete")
        result = method(prompt)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise TypeError("LLM provider generate result must be a string")
        return result

    async def stream_tokens(self, prompt: str) -> Any:
        """Forward streaming generation to the injected provider."""
        if self._cancelled:
            raise RuntimeError("LLM backend has been cancelled")
        method = self._provider_method("stream_tokens", "generate_stream")
        result = method(prompt)
        if inspect.isawaitable(result):
            return await result
        return result

    async def cancel(self) -> None:
        """Cancel provider work and reject new prompts until reset."""
        self._cancelled = True
        cancel = getattr(self.provider, "cancel", None)
        if cancel is not None:
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Prepare the adapter for a new generation session."""
        self._cancelled = False

    def _provider_method(self, *names: str):
        for name in names:
            method = getattr(self.provider, name, None)
            if callable(method):
                return method
        names_text = ", ".join(f"{name}()" for name in names)
        raise RuntimeError(f"LLM provider must expose one of: {names_text}.")


StreamingLLMAdapter = StreamingLLMBackend
