"""Hardware backend contracts isolated from the AudioStream contract."""

from abc import ABC, abstractmethod

from src.audio.frames import AudioFrame


class AudioInputBackend(ABC):
    """Low-level microphone backend returning raw PCM chunks."""

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def read(self) -> bytes | AudioFrame:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError


class AudioOutputBackend(ABC):
    """Low-level speaker backend accepting transport-ready frames."""

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError
