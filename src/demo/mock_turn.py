"""Mock turn adapter for deterministic demo event generation."""

from src.core.events.events import (
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)
from src.core.interfaces.turn import TurnAdapter


class MockTurnAdapter(TurnAdapter):
    """Generate turn events directly without processing audio."""

    def __init__(self) -> None:
        self._event_number = 0

    async def push_audio(self, audio_chunk: bytes) -> None:
        """Ignore audio; this adapter only exposes scripted event helpers."""
        return None

    def _next_event_id(self) -> str:
        self._event_number += 1
        return f"mock-turn-{self._event_number}"

    def backchannel(self, text: str = "嗯嗯") -> UserBackchannelEvent:
        return UserBackchannelEvent(
            self._next_event_id(),
            0.0,
            "mock-turn",
            {"text": text},
        )

    def interrupt(self, text: str = "等等，我想问上海") -> UserInterruptEvent:
        return UserInterruptEvent(
            self._next_event_id(),
            0.0,
            "mock-turn",
            {"text": text},
        )

    def turn_end(self) -> UserTurnEndEvent:
        return UserTurnEndEvent(
            self._next_event_id(),
            0.0,
            "mock-turn",
            {},
        )
