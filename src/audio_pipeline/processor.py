"""Audio stream, buffer, and router coordination."""

import asyncio

from src.audio.buffer import AudioBuffer
from src.audio.frames import AudioFrame
from src.audio.stream import AudioStream

from .router import AudioRouter


class AudioProcessor:
    """Move frames from an async stream through a buffer into an audio router."""

    def __init__(
        self,
        stream: AudioStream,
        buffer: AudioBuffer,
        router: AudioRouter,
    ) -> None:
        self.stream = stream
        self.buffer = buffer
        self.router = router
        self._tasks: tuple[asyncio.Task[None], asyncio.Task[None]] | None = None
        self._running = False

    @property
    def running(self) -> bool:
        """Return whether background frame forwarding is active."""
        return self._running

    async def start(self) -> None:
        """Start reading and routing frames; repeated starts are harmless."""
        if self._running:
            return
        self._running = True
        self._tasks = (
            asyncio.create_task(self._read_into_buffer()),
            asyncio.create_task(self._route_from_buffer()),
        )

    async def stop(self) -> None:
        """Stop background forwarding and cancel pending stream operations."""
        if not self._running and self._tasks is None:
            return
        self._running = False
        tasks = self._tasks
        self._tasks = None
        if tasks is None:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def process_frame(self, frame: AudioFrame) -> None:
        """Forward one frame through the router."""
        await self.router.route(frame)

    async def _read_into_buffer(self) -> None:
        while self._running:
            frame = await self.stream.read()
            await self.buffer.push(frame)

    async def _route_from_buffer(self) -> None:
        while self._running:
            frame = await self.buffer.pop()
            await self.process_frame(frame)
