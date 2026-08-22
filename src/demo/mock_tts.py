"""Fake streaming text-to-speech adapter."""

import asyncio

from src.core.interfaces.tts import TTSAdapter


class MockTTSAdapter(TTSAdapter):
    """Stream fake audio chunks and stop when interrupted."""

    def __init__(self, chunk_delay: float = 0.01):
        self.chunk_delay = chunk_delay
        self.is_speaking = False
        self.interrupted = False
        self.interrupt_count = 0
        self.audio_chunks: list[bytes] = []

    async def synthesize(self, text: str) -> None:
        """Record synthesis input without producing real audio."""
        self.audio_chunks = []
        await self.stream_audio(text)

    async def stream_audio(self, text: str) -> None:
        """Emit encoded fake chunks with a small delay between chunks."""
        self.interrupted = False
        self.is_speaking = True
        try:
            for chunk in text:
                if self.interrupted:
                    break
                await asyncio.sleep(self.chunk_delay)
                if self.interrupted:
                    break
                self.audio_chunks.append(chunk.encode("utf-8"))
        finally:
            self.is_speaking = False

    async def interrupt(self) -> None:
        """Request immediate cessation of simulated audio output."""
        self.interrupted = True
        self.interrupt_count += 1
        self.is_speaking = False
