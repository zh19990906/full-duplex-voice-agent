"""Minimal per-session composition root for the realtime audio path."""

from .audio_ingress import AudioIngress
from .identifiers import GenerationClock


class RealtimeSessionRuntime:
    """Own one session identity, generation epoch, and audio ingress lifecycle."""

    def __init__(self, session_id: str, *, ingress: AudioIngress | None = None) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = session_id
        self.generation_clock = GenerationClock()
        self.ingress = ingress if ingress is not None else AudioIngress()
        self._closed = False

    @property
    def generation_epoch(self) -> int:
        """Return the active generation epoch for this session."""
        return self.generation_clock.current

    @property
    def closed(self) -> bool:
        """Return whether the session runtime has been closed."""
        return self._closed

    def advance_generation(self) -> int:
        """Invalidate current response work and return the replacement epoch."""
        return self.generation_clock.advance()

    async def push_audio(self, frame: bytes) -> None:
        """Delegate a browser PCM frame to this session's ingress."""
        if self._closed:
            raise RuntimeError("realtime session runtime is closed")
        await self.ingress.push(frame)

    async def flush(self) -> None:
        """Wait for this session's accepted audio to reach all consumers."""
        await self.ingress.flush()

    async def close(self) -> None:
        """Close the ingress and make the session unavailable for more audio."""
        if self._closed:
            return
        self._closed = True
        await self.ingress.close()
