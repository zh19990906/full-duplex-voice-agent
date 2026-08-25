"""Epoch-fenced LLM token to stable TTS segment streaming."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import time
from typing import Any

from src.llm_runtime.stream import TokenChunk

from .cancellation import CancellationToken
from .identifiers import GenerationClock, IdentifierAllocator
from .text_segmenter import LanguageAwareTextSegmenter, TextSegment


TokenPublisher = Callable[[TokenChunk], Awaitable[None] | None]


@dataclass(frozen=True)
class ResponsePipelineResult:
    """Terminal state for one response generation attempt."""

    response_id: str
    generation_epoch: int
    text: str
    cancelled: bool = False
    stale: bool = False


@dataclass(frozen=True)
class ResponseStreamEnd:
    """Terminal queue control item for one completed current response."""

    response_id: str
    generation_epoch: int
    final_segment_id: int | None
    status: str = "completed"

    def __post_init__(self) -> None:
        if not isinstance(self.response_id, str) or not self.response_id:
            raise ValueError("response_id must be a nonempty string")
        if isinstance(self.generation_epoch, bool) or not isinstance(
            self.generation_epoch, int
        ):
            raise TypeError("generation_epoch must be an integer")
        if self.generation_epoch < 0:
            raise ValueError("generation_epoch must be nonnegative")
        if self.final_segment_id is not None:
            if isinstance(self.final_segment_id, bool) or not isinstance(
                self.final_segment_id, int
            ):
                raise TypeError("final_segment_id must be an integer or None")
            if self.final_segment_id < 0:
                raise ValueError("final_segment_id must be nonnegative")
        if self.status != "completed":
            raise ValueError("ResponseStreamEnd status must be 'completed'")


class RealtimeResponsePipeline:
    """Serialize token generation and enqueue only current stable text."""

    def __init__(
        self,
        *,
        llm: Any,
        segment_queue: Any,
        generation_clock: GenerationClock,
        segmenter: LanguageAwareTextSegmenter | None = None,
        cancellation_token: CancellationToken | None = None,
        response_id_factory: Callable[[], str] | None = None,
        on_token: TokenPublisher | None = None,
        capacity_poll_seconds: float = 0.01,
    ) -> None:
        if not callable(getattr(llm, "stream_tokens", None)):
            raise TypeError("llm must expose stream_tokens(prompt)")
        if not callable(getattr(segment_queue, "put_nowait", None)):
            raise TypeError("segment_queue must expose put_nowait(item)")
        if not isinstance(generation_clock, GenerationClock):
            raise TypeError("generation_clock must be a GenerationClock")
        if segmenter is not None and not isinstance(segmenter, LanguageAwareTextSegmenter):
            raise TypeError("segmenter must be a LanguageAwareTextSegmenter")
        if cancellation_token is not None and not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken")
        if not isinstance(capacity_poll_seconds, (int, float)) or capacity_poll_seconds <= 0:
            raise ValueError("capacity_poll_seconds must be positive")
        self.llm = llm
        self.segment_queue = segment_queue
        self.generation_clock = generation_clock
        self.segmenter = segmenter or LanguageAwareTextSegmenter()
        self.cancellation_token = cancellation_token
        self.on_token = on_token
        self.capacity_poll_seconds = float(capacity_poll_seconds)
        allocator = IdentifierAllocator()
        self._response_id_factory = response_id_factory or allocator.next_response_id
        self._run_lock = asyncio.Lock()
        self._provider_needs_reset = False

    async def run(self, prompt: str) -> ResponsePipelineResult:
        """Generate one response without advancing the caller-owned epoch clock."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        async with self._run_lock:
            if self._provider_needs_reset and not self._externally_cancelled():
                await _reset_provider(self.llm)
                self._provider_needs_reset = False
            response_id = self._next_response_id()
            epoch = self.generation_clock.current
            self.segmenter.reset()
            pieces: list[str] = []
            final_segment_id: int | None = None
            provider_cancelled = False

            async def stop_provider() -> None:
                nonlocal provider_cancelled
                if provider_cancelled:
                    return
                provider_cancelled = True
                self._provider_needs_reset = True
                await _interrupt_provider(self.llm)

            try:
                if not self._is_current(epoch):
                    await stop_provider()
                    return self._result(response_id, epoch, pieces, stale=True)
                stream = self.llm.stream_tokens(prompt)
                if inspect.isawaitable(stream):
                    stream = await stream
                async for raw_token in stream:
                    if not self._is_current(epoch):
                        await stop_provider()
                        return self._result(response_id, epoch, pieces)
                    token = _normalize_token(raw_token)
                    if token.text:
                        tagged = TokenChunk(
                            token.chunk_id,
                            token.text,
                            token.timestamp,
                            token.is_final,
                            response_id,
                            epoch,
                        )
                        if not self._is_current(epoch):
                            await stop_provider()
                            return self._result(response_id, epoch, pieces)
                        if not await self._publish(tagged, epoch):
                            await stop_provider()
                            return self._result(response_id, epoch, pieces)
                        if not self._is_current(epoch):
                            await stop_provider()
                            return self._result(response_id, epoch, pieces)
                        pieces.append(token.text)
                        segments = (
                            self.segmenter.flush(token.text)
                            if token.is_final
                            else self.segmenter.push(token.text)
                        )
                        for segment in segments:
                            if not await self._enqueue(segment, response_id, epoch):
                                await stop_provider()
                                return self._result(response_id, epoch, pieces)
                            final_segment_id = segment.segment_id
                    if token.is_final:
                        break
                if not self._is_current(epoch):
                    await stop_provider()
                    return self._result(response_id, epoch, pieces)
                for segment in self.segmenter.flush():
                    if not await self._enqueue(segment, response_id, epoch):
                        await stop_provider()
                        return self._result(response_id, epoch, pieces)
                    final_segment_id = segment.segment_id
                terminal = ResponseStreamEnd(response_id, epoch, final_segment_id)
                if not await self._enqueue_item(terminal, epoch):
                    await stop_provider()
                    return self._result(response_id, epoch, pieces)
                return self._result(response_id, epoch, pieces, cancelled=False, stale=False)
            except asyncio.CancelledError:
                await stop_provider()
                self.segmenter.reset()
                raise
            finally:
                if not self._is_current(epoch):
                    self.segmenter.reset()

    async def _publish(self, token: TokenChunk, epoch: int) -> bool:
        if not self._is_current(epoch):
            return False
        if self.on_token is None:
            return self._is_current(epoch)
        result = self.on_token(token)
        if inspect.isawaitable(result):
            await result
        return self._is_current(epoch)

    async def _enqueue(self, segment: TextSegment, response_id: str, epoch: int) -> bool:
        if not self._is_current(epoch):
            return False
        tagged = TextSegment(
            segment.segment_id,
            segment.text,
            segment.is_final,
            response_id,
            epoch,
        )
        return await self._enqueue_item(tagged, epoch)

    async def _enqueue_item(self, item: TextSegment | ResponseStreamEnd, epoch: int) -> bool:
        while self._is_current(epoch):
            try:
                self.segment_queue.put_nowait(item)
                return self._is_current(epoch)
            except asyncio.QueueFull:
                await asyncio.sleep(self.capacity_poll_seconds)
        return False

    def _is_current(self, epoch: int) -> bool:
        return self.generation_clock.is_current(epoch) and not (
            self._externally_cancelled()
        )

    def _externally_cancelled(self) -> bool:
        return self.cancellation_token is not None and self.cancellation_token.is_cancelled()

    def _next_response_id(self) -> str:
        response_id = self._response_id_factory()
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("response_id_factory must return a nonempty string")
        return response_id

    def _result(
        self,
        response_id: str,
        epoch: int,
        pieces: list[str],
        *,
        cancelled: bool | None = None,
        stale: bool | None = None,
    ) -> ResponsePipelineResult:
        was_cancelled = self._externally_cancelled()
        was_stale = not self.generation_clock.is_current(epoch)
        return ResponsePipelineResult(
            response_id=response_id,
            generation_epoch=epoch,
            text="".join(pieces),
            cancelled=was_cancelled if cancelled is None else cancelled,
            stale=was_stale if stale is None else stale,
        )


def _normalize_token(value: Any) -> TokenChunk:
    if isinstance(value, TokenChunk):
        return value
    if isinstance(value, str):
        return TokenChunk("token", value, time.time(), False)
    raise TypeError("LLM token must be a string or TokenChunk")


async def _interrupt_provider(provider: Any) -> None:
    method = getattr(provider, "cancel", None) or getattr(provider, "interrupt", None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


async def _reset_provider(provider: Any) -> None:
    method = getattr(provider, "reset", None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


__all__ = ["RealtimeResponsePipeline", "ResponsePipelineResult", "ResponseStreamEnd"]
