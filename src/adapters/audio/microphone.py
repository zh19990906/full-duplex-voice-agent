"""Microphone AudioStream adapter over an injected device backend."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from src.audio.frames import AudioFrame
from src.audio.stream import AudioStream


class MicrophoneAdapter(AudioStream):
    """Convert backend input chunks into ``AudioFrame`` objects."""

    def __init__(
        self,
        backend: Any,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        if backend is None:
            raise ValueError("a microphone backend must be supplied")
        self.backend = backend
        self.sample_rate = sample_rate
        self.channels = channels
        self._started = False
        self._stopped = False
        self._sequence = 0

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

    async def read(self) -> AudioFrame:
        if self._stopped:
            raise RuntimeError("microphone has been stopped")
        if not self._started:
            await self.start()
        result = self.backend.read()
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, AudioFrame):
            return result
        if not isinstance(result, bytes):
            raise TypeError("microphone backend must return bytes or AudioFrame")
        self._sequence += 1
        return AudioFrame(
            frame_id=f"microphone-{self._sequence}-{uuid4().hex[:8]}",
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            data=result,
        )

    async def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError("MicrophoneAdapter is input-only")

    async def stop(self) -> None:
        self._stopped = True
        if not self._started:
            method = getattr(self.backend, "stop", None)
            if callable(method):
                result = method()
                if hasattr(result, "__await__"):
                    await result
            return
        self._started = False
        method = getattr(self.backend, "stop", None)
        if callable(method):
            result = method()
            if hasattr(result, "__await__"):
                await result

    def __aiter__(self) -> "MicrophoneAdapter":
        return self

    async def __anext__(self) -> AudioFrame:
        if self._stopped:
            raise StopAsyncIteration
        return await self.read()
