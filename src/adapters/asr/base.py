"""Model-independent streaming ASR adapter contract."""

from abc import ABC, abstractmethod


class BaseASRAdapter(ABC):
    """Receive audio chunks and expose future partial transcription output."""

    @abstractmethod
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Accept one audio chunk for streaming speech recognition."""
        raise NotImplementedError
