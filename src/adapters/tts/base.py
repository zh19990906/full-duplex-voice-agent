"""Model-independent streaming TTS adapter contract."""

from abc import ABC, abstractmethod


class BaseTTSAdapter(ABC):
    """Define synthesis, audio streaming, and interruption boundaries."""

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
