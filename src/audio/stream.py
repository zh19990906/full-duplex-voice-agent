"""Abstract asynchronous audio stream interface."""

from abc import ABC, abstractmethod

from .frames import AudioFrame


class AudioStream(ABC):
    """Receive and send audio frames without hardware or network behavior."""

    @abstractmethod
    async def read(self) -> AudioFrame:
        """Read the next audio frame from the stream."""
        raise NotImplementedError

    @abstractmethod
    async def write(self, frame: AudioFrame) -> None:
        """Write one audio frame to the stream."""
        raise NotImplementedError
