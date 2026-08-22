"""FIFO asynchronous audio frame buffer."""

import asyncio

from .frames import AudioFrame


class AudioBuffer:
    """Buffer audio frames using an asyncio FIFO queue."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=maxsize)

    async def push(self, frame: AudioFrame) -> None:
        """Append a frame, waiting when a bounded buffer is full."""
        await self._queue.put(frame)

    async def pop(self) -> AudioFrame:
        """Remove and return the oldest frame, waiting when empty."""
        return await self._queue.get()

    def empty(self) -> bool:
        """Return whether the buffer currently has no frames."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the current number of buffered frames."""
        return self._queue.qsize()
