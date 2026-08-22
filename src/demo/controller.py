"""Minimal conversation controller for the vertical-slice demo."""

import asyncio

from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
)
from src.core.interfaces.llm import LLMAdapter
from src.core.interfaces.tts import TTSAdapter


class ConversationController:
    """Keep generation on backchannel and cancel output on interruption."""

    def __init__(self, llm: LLMAdapter, tts: TTSAdapter) -> None:
        self.llm = llm
        self.tts = tts
        self.backchannel_count = 0
        self.last_interrupt: UserInterruptEvent | None = None
        self._response_task: asyncio.Task[None] | None = None

    async def start_response(self, prompt: str) -> None:
        """Start the mock generation-to-speech response flow."""
        await self.stop_response()
        self._response_task = asyncio.create_task(self._run_response(prompt))

    async def _run_response(self, prompt: str) -> None:
        response = await self.llm.generate(prompt)
        if response and not getattr(self.llm, "cancelled", False):
            await self.tts.stream_audio(response)

    async def handle_event(self, event: BaseEvent) -> None:
        """Apply only the demo's backchannel and interruption decisions."""
        if isinstance(event, UserBackchannelEvent):
            self.backchannel_count += 1
            return
        if isinstance(event, UserInterruptEvent):
            await self.tts.interrupt()
            await self.llm.cancel()
            self.last_interrupt = event
            if self._response_task is not None:
                await self._response_task
                self._response_task = None

    async def stop_response(self) -> None:
        """Stop any active demo response and wait for its task."""
        if self._response_task is None:
            return
        await self.tts.interrupt()
        await self.llm.cancel()
        await self._response_task
        self._response_task = None
