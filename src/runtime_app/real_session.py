"""Real model session orchestration shared by API and WebSocket transports."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from src.llm_runtime.stream import TokenChunk
from src.realtime.cancellation import CancellationToken
from src.realtime.identifiers import GenerationClock, IdentifierAllocator
from src.realtime.response_pipeline import RealtimeResponsePipeline, ResponseStreamEnd
from src.realtime.text_segmenter import LanguageAwareTextSegmenter, TextSegment
from src.tts_runtime.stream import AudioChunk


Publish = Callable[[Any], Awaitable[None] | None]
_QUEUE_STOP = object()


class RealModelSession:
    """Connect a streaming LLM and TTS provider for one user session."""

    def __init__(
        self,
        llm: Any,
        tts: Any,
        publish: Publish,
        *,
        prompt_builder: Callable[[str], str] | None = None,
        segmenter: LanguageAwareTextSegmenter | None = None,
        tts_queue_capacity: int = 2,
    ) -> None:
        if segmenter is not None and not isinstance(segmenter, LanguageAwareTextSegmenter):
            raise TypeError("segmenter must be a LanguageAwareTextSegmenter")
        if isinstance(tts_queue_capacity, bool) or not isinstance(tts_queue_capacity, int):
            raise TypeError("tts_queue_capacity must be an integer")
        if tts_queue_capacity <= 0:
            raise ValueError("tts_queue_capacity must be positive")
        self.llm = llm
        self.tts = tts
        self.publish = publish
        self.prompt_builder = prompt_builder or (lambda text: text)
        self.generation_clock = GenerationClock()
        self._identifiers = IdentifierAllocator()
        self._segmenter = segmenter or LanguageAwareTextSegmenter()
        self._tts_queue_capacity = tts_queue_capacity
        self._run_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._active_token: CancellationToken | None = None
        self._active_epoch: int | None = None
        self._active_invalidated = False
        self._active_consumer_task: asyncio.Task[None] | None = None
        self._active_tts_start_task: asyncio.Task[Any] | None = None
        self._active_tts_request_id: str | None = None
        self.last_response_end: ResponseStreamEnd | None = None
        self._interrupted = False

    async def run(self, text: str) -> str:
        """Stream one LLM response followed by its TTS audio."""

        async with self._run_lock:
            await self._reset_provider(self.llm)
            await self._reset_provider(self.tts)
            epoch, token = await self._begin_run()
            queue: asyncio.Queue[TextSegment | object] = asyncio.Queue(
                maxsize=self._tts_queue_capacity
            )
            pipeline = RealtimeResponsePipeline(
                llm=self.llm,
                segment_queue=queue,
                generation_clock=self.generation_clock,
                segmenter=self._segmenter,
                cancellation_token=token,
                response_id_factory=self._identifiers.next_response_id,
                on_token=self._emit,
            )
            consumer = asyncio.create_task(self._consume_segments(queue, epoch, token))
            await self._set_active_consumer(epoch, consumer)
            try:
                result = await pipeline.run(self.prompt_builder(text))
                if not self._is_current(epoch, token):
                    await self._await_interrupted_consumer(consumer)
                    return result.text
                if consumer.done():
                    await consumer
                await consumer
                return result.text
            except BaseException:
                token.cancel()
                await self._interrupt_provider(self.llm)
                await self._cancel_consumer(consumer)
                raise
            finally:
                await self._finish_run(epoch, token, consumer)

    async def interrupt(self) -> None:
        """Cancel both generation stages and prevent stale audio publication."""

        async with self._state_lock:
            self._interrupted = True
            token = self._active_token
            if token is not None:
                token.cancel()
            if self._active_epoch is not None and not self._active_invalidated:
                self.generation_clock.advance()
                self._active_invalidated = True
            consumer = self._active_consumer_task
            startup = self._active_tts_start_task
            tts_request_id = self._active_tts_request_id
        if startup is not None and not startup.done():
            startup.cancel()
        if consumer is not None and not consumer.done():
            consumer.cancel()
        if tts_request_id is not None:
            await self._cancel_tts_request(tts_request_id)
        for provider in (self.llm, self.tts):
            await self._interrupt_provider(provider)

    async def _begin_run(self) -> tuple[int, CancellationToken]:
        async with self._state_lock:
            self._interrupted = False
            epoch = self.generation_clock.advance()
            token = CancellationToken()
            self._active_token = token
            self._active_epoch = epoch
            self._active_invalidated = False
            return epoch, token

    async def _finish_run(
        self,
        epoch: int,
        token: CancellationToken,
        consumer: asyncio.Task[None],
    ) -> None:
        async with self._state_lock:
            if self._active_consumer_task is consumer:
                self._active_consumer_task = None
            if self._active_tts_start_task is not None and self._active_tts_start_task.done():
                self._active_tts_start_task = None
            if self._active_epoch == epoch:
                self._active_tts_request_id = None
            if self._active_epoch == epoch and self._active_token is token:
                self._active_token = None
                self._active_epoch = None
                self._active_invalidated = False

    async def _set_active_consumer(
        self, epoch: int, consumer: asyncio.Task[None]
    ) -> None:
        async with self._state_lock:
            if self._active_epoch == epoch:
                self._active_consumer_task = consumer

    async def _consume_segments(
        self,
        queue: asyncio.Queue[TextSegment | object],
        epoch: int,
        cancellation_token: CancellationToken,
    ) -> None:
        try:
            while True:
                item = await queue.get()
                try:
                    if item is _QUEUE_STOP:
                        return
                    if isinstance(item, ResponseStreamEnd):
                        if self._is_current(epoch, cancellation_token):
                            await self._record_terminal(item)
                        return
                    assert isinstance(item, TextSegment)
                    if not self._is_current(epoch, cancellation_token):
                        continue
                    request_id = self._tts_request_id(item)
                    try:
                        audio = await self._await_tts_start(item, epoch, request_id)
                        if not self._is_current(epoch, cancellation_token):
                            continue
                        async for chunk in audio:
                            if not self._is_current(epoch, cancellation_token):
                                break
                            await self._emit(self._tag_audio(chunk, item, request_id))
                            if not self._is_current(epoch, cancellation_token):
                                break
                    finally:
                        await self._clear_active_tts_request(epoch, request_id)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            if not self._is_current(epoch, cancellation_token):
                return
            raise
        except BaseException:
            cancellation_token.cancel()
            await self._interrupt_provider(self.llm)
            raise

    async def _await_tts_start(
        self, segment: TextSegment, epoch: int, request_id: str
    ) -> Any:
        task = asyncio.create_task(self._start_tts(segment, request_id))
        await self._set_active_tts_start(epoch, task, request_id)
        try:
            return await task
        finally:
            await self._clear_active_tts_start(epoch, task)

    async def _set_active_tts_start(
        self, epoch: int, task: asyncio.Task[Any], request_id: str
    ) -> None:
        async with self._state_lock:
            if self._active_epoch == epoch:
                self._active_tts_start_task = task
                self._active_tts_request_id = request_id

    async def _clear_active_tts_start(self, epoch: int, task: asyncio.Task[Any]) -> None:
        async with self._state_lock:
            if self._active_epoch == epoch and self._active_tts_start_task is task:
                self._active_tts_start_task = None

    async def _clear_active_tts_request(self, epoch: int, request_id: str) -> None:
        async with self._state_lock:
            if self._active_epoch == epoch and self._active_tts_request_id == request_id:
                self._active_tts_request_id = None

    async def _record_terminal(self, marker: ResponseStreamEnd) -> None:
        async with self._state_lock:
            if self.generation_clock.is_current(marker.generation_epoch):
                self.last_response_end = marker

    async def _await_interrupted_consumer(self, consumer: asyncio.Task[None]) -> None:
        try:
            await consumer
        except asyncio.CancelledError:
            return

    @staticmethod
    async def _cancel_consumer(consumer: asyncio.Task[None]) -> None:
        if not consumer.done():
            consumer.cancel()
        try:
            await consumer
        except (asyncio.CancelledError, Exception):
            return

    async def _start_tts(self, segment: TextSegment, request_id: str) -> Any:
        options = {
            "request_id": request_id,
            "response_id": segment.response_id,
            "generation_epoch": segment.generation_epoch,
            "segment_id": segment.segment_id,
        }
        audio = self._call_supported(self.tts.stream_audio, segment.text, **options)
        if inspect.isawaitable(audio):
            return await audio
        return audio

    @staticmethod
    def _tts_request_id(segment: TextSegment) -> str:
        if segment.response_id is None or segment.generation_epoch is None:
            raise RuntimeError("TTS segment is missing realtime identity")
        return f"{segment.response_id}:{segment.generation_epoch}:{segment.segment_id}"

    @staticmethod
    def _tag_audio(chunk: Any, segment: TextSegment, request_id: str) -> AudioChunk:
        if not isinstance(chunk, AudioChunk):
            raise TypeError("TTS provider must yield AudioChunk")
        expected = {
            "request_id": request_id,
            "response_id": segment.response_id,
            "generation_epoch": segment.generation_epoch,
            "segment_id": segment.segment_id,
        }
        for name, value in expected.items():
            actual = getattr(chunk, name)
            if actual is not None and actual != value:
                raise RuntimeError(f"TTS provider returned a mismatched {name}")
        return replace(chunk, **expected)

    async def _cancel_tts_request(self, request_id: str) -> None:
        method = getattr(self.tts, "cancel", None) or getattr(self.tts, "interrupt", None)
        if not callable(method):
            return
        result = self._call_supported(method, request_id)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _call_supported(method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return method(*args, **kwargs)
        positional_count = sum(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        supported_args = args[:positional_count]
        return method(
            *supported_args,
            **{key: value for key, value in kwargs.items() if key in signature.parameters},
        )

    def _is_current(self, epoch: int, token: CancellationToken) -> bool:
        return self.generation_clock.is_current(epoch) and not token.is_cancelled()

    @staticmethod
    async def _interrupt_provider(provider: Any) -> None:
        method = getattr(provider, "interrupt", None) or getattr(provider, "cancel", None)
        if callable(method):
            result = method()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    async def _reset_provider(provider: Any) -> None:
        method = getattr(provider, "reset", None)
        if callable(method):
            result = method()
            if inspect.isawaitable(result):
                await result

    async def _emit(self, value: Any) -> None:
        result = self.publish(value)
        if inspect.isawaitable(result):
            await result


__all__ = ["RealModelSession"]
