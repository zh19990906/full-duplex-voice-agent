"""Real model session orchestration shared by API and WebSocket transports."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from src.llm_runtime.stream import TokenChunk


Publish = Callable[[Any], Awaitable[None] | None]


class RealModelSession:
    """Connect a streaming LLM and TTS provider for one user session."""

    def __init__(
        self,
        llm: Any,
        tts: Any,
        publish: Publish,
        *,
        prompt_builder: Callable[[str], str] | None = None,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.publish = publish
        self.prompt_builder = prompt_builder or (lambda text: text)
        self._interrupted = False

    async def run(self, text: str) -> str:
        """Stream one LLM response followed by its TTS audio."""

        self._interrupted = False
        pieces: list[str] = []
        async for token in self.llm.stream_tokens(self.prompt_builder(text)):
            if self._interrupted:
                return "".join(pieces)
            pieces.append(token.text if isinstance(token, TokenChunk) else str(token))
            await self._emit(token)

        response = "".join(pieces).strip()
        if not response or self._interrupted:
            return response
        audio = await self.tts.stream_audio(response)
        async for chunk in audio:
            if self._interrupted:
                return response
            await self._emit(chunk)
        return response

    async def interrupt(self) -> None:
        """Cancel both generation stages and prevent stale audio publication."""

        self._interrupted = True
        for provider in (self.llm, self.tts):
            method = getattr(provider, "interrupt", None) or getattr(provider, "cancel", None)
            if callable(method):
                result = method()
                if inspect.isawaitable(result):
                    await result

    async def _emit(self, value: Any) -> None:
        result = self.publish(value)
        if inspect.isawaitable(result):
            await result


__all__ = ["RealModelSession"]
