"""Speaker AudioStream adapter over an injected device backend."""

from __future__ import annotations

import asyncio
from typing import Any

from src.audio.frames import AudioFrame
from src.audio.stream import AudioStream


class SpeakerAdapter(AudioStream):
    """Write frames to a backend sequentially, preserving playback order."""

    def __init__(self, backend: Any) -> None:
        if backend is None:
            raise ValueError("a speaker backend must be supplied")
        self.backend = backend
        self._started = False
        self._stopped = False
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        method = getattr(self.backend, "start", None)
        if callable(method):
            result = method()
            if hasattr(result, "__await__"):
                await result
        self._started = True
        self._stopped = False

    async def write(self, frame: AudioFrame) -> None:
        if not isinstance(frame, AudioFrame):
            raise TypeError("speaker expects an AudioFrame")
        if self._stopped:
            raise RuntimeError("speaker has been stopped")
        if not self._started:
            await self.start()
        async with self._write_lock:
            if not self._started:
                raise RuntimeError("speaker has been stopped")
            result = self.backend.write(frame)
            if hasattr(result, "__await__"):
                await result

    async def stop(self) -> None:
        self._stopped = True
        if not self._started:
            return
        self._started = False
        method = getattr(self.backend, "stop", None)
        if callable(method):
            result = method()
            if hasattr(result, "__await__"):
                await result

    async def read(self) -> AudioFrame:
        raise NotImplementedError("SpeakerAdapter is output-only")
