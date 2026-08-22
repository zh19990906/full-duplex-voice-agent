"""Optional sounddevice backends isolated from AudioStream adapters.

The dependency is imported only when this backend is explicitly selected by
configuration. Tests and other deployments can inject any backend instead.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any

from src.audio.frames import AudioFrame


def _sounddevice_module() -> Any:
    try:
        import sounddevice  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice backend selected but the optional 'sounddevice' package is unavailable"
        ) from exc
    return sounddevice


class SoundDeviceInputBackend:
    """Raw PCM microphone backend using a sounddevice callback queue."""

    def __init__(self, sample_rate: int, channels: int, device: Any = None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: Any | None = None
        self._chunks: queue.Queue[bytes] = queue.Queue()

    async def start(self) -> None:
        sounddevice = _sounddevice_module()

        def callback(indata, frames, callback_time, status) -> None:
            self._chunks.put(bytes(indata))

        self._stream = sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    async def read(self) -> bytes:
        return await asyncio.to_thread(self._chunks.get)

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SoundDeviceOutputBackend:
    """Raw PCM speaker backend using sounddevice's output stream."""

    def __init__(self, sample_rate: int, channels: int, device: Any = None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: Any | None = None

    async def start(self) -> None:
        sounddevice = _sounddevice_module()
        self._stream = sounddevice.RawOutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            dtype="int16",
        )
        self._stream.start()

    async def write(self, frame: AudioFrame) -> None:
        if self._stream is None:
            raise RuntimeError("sounddevice output backend has not been started")
        await asyncio.to_thread(self._stream.write, frame.data)

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
