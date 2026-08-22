"""Stateless audio frame routing."""

from src.audio.frames import AudioFrame
from src.core.interfaces.turn import TurnAdapter


class AudioRouter:
    """Forward raw frame bytes to a turn adapter without interpretation."""

    def __init__(self, turn_adapter: TurnAdapter) -> None:
        self.turn_adapter = turn_adapter

    async def route(self, frame: AudioFrame) -> None:
        """Send the original audio bytes from ``frame`` to the adapter."""
        await self.turn_adapter.push_audio(frame.data)
