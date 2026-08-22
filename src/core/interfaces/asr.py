"""Streaming speech recognition adapter contract."""

from abc import ABC, abstractmethod


class ASRAdapter(ABC):
    """Receive audio and expose partial transcript output."""

    @abstractmethod
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Accept one audio chunk for streaming recognition."""
        raise NotImplementedError
