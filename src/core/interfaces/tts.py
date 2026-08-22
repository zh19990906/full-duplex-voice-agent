"""Streaming speech synthesis adapter contract."""

from abc import ABC, abstractmethod


class TTSAdapter(ABC):
    """Synthesize text and support immediate interruption."""

    @abstractmethod
    async def synthesize(self, text: str) -> None:
        """Prepare speech synthesis for text."""
        raise NotImplementedError

    @abstractmethod
    async def stream_audio(self, text: str) -> None:
        """Start streaming synthesized audio for text."""
        raise NotImplementedError

    @abstractmethod
    async def interrupt(self) -> None:
        """Immediately interrupt active speech output."""
        raise NotImplementedError
