"""Turn detection adapter contract."""

from abc import ABC, abstractmethod


class TurnAdapter(ABC):
    """Receive audio and expose conversational turn events."""

    @abstractmethod
    async def push_audio(self, audio_chunk: bytes) -> None:
        """Accept one audio chunk for turn detection."""
        raise NotImplementedError
