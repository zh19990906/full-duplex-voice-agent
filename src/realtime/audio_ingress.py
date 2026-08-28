"""Ordered, bounded fan-out for V1 browser PCM audio frames."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import math
import struct
from typing import Any

from .protocol import AudioFrameHeader, decode_audio_frame


@dataclass(frozen=True)
class RealtimeAudioFrame:
    """One decoded browser frame, preserving its V1 header and PCM payload."""

    header: AudioFrameHeader
    pcm: bytes

    @property
    def sequence(self) -> int:
        return self.header.sequence

    @property
    def capture_timestamp(self) -> float:
        return self.header.capture_timestamp


@dataclass(frozen=True)
class AudioActivityCandidate:
    """A data-only PCM energy observation for later fusion and policy."""

    sequence: int
    capture_timestamp: float
    active: bool
    rms: float


@dataclass
class AudioIngressMetrics:
    """Explicit loss and ordering metrics for the ingress lifecycle."""

    duplicate_frames: int = 0
    jitter_buffer_overflow_frames: int = 0
    consumer_overflow_frames: dict[str, int] = field(default_factory=dict)
    missing_sequence_ranges: list[tuple[int, int]] = field(default_factory=list)


class AudioIngressConsumerError(RuntimeError):
    """A downstream consumer failed after audio was accepted by ingress."""


AudioConsumer = Callable[[RealtimeAudioFrame], Awaitable[None] | None]
ActivityConsumer = Callable[[AudioActivityCandidate], Awaitable[None] | None]


class AudioIngress:
    """Decode browser frames once, order them, and fan them out independently.

    The jitter buffer rejects the newest future frame when full. Each consumer
    queue independently drops its newest frame when full, so one slow model
    cannot block another model's real-time path.
    """

    _STOP = object()

    def __init__(
        self,
        *,
        asr_consumer: AudioConsumer | None = None,
        turn_consumer: AudioConsumer | None = None,
        activity_consumer: ActivityConsumer | None = None,
        jitter_buffer_size: int = 32,
        consumer_queue_size: int = 32,
        activity_threshold: float = 500.0,
    ) -> None:
        if jitter_buffer_size < 1:
            raise ValueError("jitter_buffer_size must be at least one")
        if consumer_queue_size < 1:
            raise ValueError("consumer_queue_size must be at least one")
        if activity_threshold < 0:
            raise ValueError("activity_threshold must not be negative")

        self._jitter_buffer_size = jitter_buffer_size
        self._activity_threshold = activity_threshold
        self._next_sequence = 0
        self._buffer: dict[int, RealtimeAudioFrame] = {}
        self._closed = False
        self._lock = asyncio.Lock()
        self.metrics = AudioIngressMetrics()
        self._last_activity_candidate: AudioActivityCandidate | None = None
        self._worker_errors: dict[str, BaseException] = {}
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._consumers: dict[str, Callable[[Any], Awaitable[None] | None]] = {}
        self._native_async_consumers: dict[str, bool] = {}
        self._close_task: asyncio.Task[None] | None = None

        self._register_consumer("asr", asr_consumer, consumer_queue_size)
        self._register_consumer("turn", turn_consumer, consumer_queue_size)
        self._register_consumer("activity", activity_consumer, consumer_queue_size)

    @property
    def next_sequence(self) -> int:
        """Return the next expected browser frame sequence."""
        return self._next_sequence

    @property
    def last_activity_candidate(self) -> AudioActivityCandidate | None:
        """Return the latest data-only PCM activity observation."""
        return self._last_activity_candidate

    @property
    def worker_errors(self) -> dict[str, BaseException]:
        """Return the first failure recorded for each named consumer."""
        return dict(self._worker_errors)

    @property
    def closed(self) -> bool:
        """Return whether no additional browser frames may be accepted."""
        return self._closed

    async def push(self, frame: bytes) -> None:
        """Decode and accept one V1 browser frame without blocking consumers."""
        header, pcm = decode_audio_frame(frame)
        await self.push_decoded(RealtimeAudioFrame(header=header, pcm=pcm))

    async def push_decoded(
        self,
        decoded: RealtimeAudioFrame,
        *,
        yield_consumers: bool = True,
    ) -> None:
        """Accept one already-decoded V1 browser frame."""
        if not isinstance(decoded, RealtimeAudioFrame):
            raise TypeError("decoded must be RealtimeAudioFrame")

        async with self._lock:
            if self._closed:
                raise RuntimeError("audio ingress is closed")
            if decoded.sequence < self._next_sequence or decoded.sequence in self._buffer:
                self.metrics.duplicate_frames += 1
                raise ValueError(f"duplicate audio frame sequence: {decoded.sequence}")

            if decoded.sequence > self._next_sequence:
                if len(self._buffer) >= self._jitter_buffer_size:
                    self.metrics.jitter_buffer_overflow_frames += 1
                    raise OverflowError("jitter buffer is full")
                self._buffer[decoded.sequence] = decoded
                return

            self._deliver_contiguous(decoded)
        # A continuous producer must not starve independent consumer workers.
        # The server may defer this single yield until after it publishes the
        # frame-accepted acknowledgement.
        if yield_consumers:
            await asyncio.sleep(0)

    async def flush(self) -> None:
        """Drain buffered frames in sorted order, record gaps, and await workers."""
        async with self._lock:
            for sequence in sorted(self._buffer):
                if sequence > self._next_sequence:
                    self.metrics.missing_sequence_ranges.append(
                        (self._next_sequence, sequence - 1)
                    )
                self._deliver_frame(self._buffer[sequence])
                self._next_sequence = sequence + 1
            self._buffer.clear()
            queues = tuple(self._queues.values())

        for queue in queues:
            await queue.join()
        self._raise_worker_error()

    async def close(self) -> None:
        """Atomically reject new frames, drain accepted work, and stop workers."""
        async with self._lock:
            if self._close_task is None:
                # This transition happens before draining, so every frame either
                # entered the ingress before close or is rejected by push().
                self._closed = True
                self._close_task = asyncio.create_task(
                    self._close_impl(), name="audio-ingress-close"
                )
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_impl(self) -> None:
        try:
            await self.flush()
        finally:
            async with self._lock:
                worker_pairs = tuple(
                    (self._queues[name], worker)
                    for name, worker in self._workers.items()
                    if not worker.done()
                )

            for queue, _ in worker_pairs:
                await queue.put(self._STOP)
            if worker_pairs:
                await asyncio.gather(*(worker for _, worker in worker_pairs))

    def _register_consumer(
        self,
        name: str,
        consumer: Callable[[Any], Awaitable[None] | None] | None,
        queue_size: int,
    ) -> None:
        if consumer is None:
            return
        self._consumers[name] = consumer
        self._native_async_consumers[name] = _is_native_async_callable(consumer)
        self._queues[name] = asyncio.Queue(maxsize=queue_size)
        self.metrics.consumer_overflow_frames[name] = 0

    def _deliver_contiguous(self, frame: RealtimeAudioFrame) -> None:
        self._deliver_frame(frame)
        self._next_sequence += 1
        while self._next_sequence in self._buffer:
            contiguous = self._buffer.pop(self._next_sequence)
            self._deliver_frame(contiguous)
            self._next_sequence += 1

    def _deliver_frame(self, frame: RealtimeAudioFrame) -> None:
        self._enqueue("asr", frame)
        self._enqueue("turn", frame)

        rms = _pcm16_rms(frame.pcm)
        candidate = AudioActivityCandidate(
            sequence=frame.sequence,
            capture_timestamp=frame.capture_timestamp,
            rms=rms,
            active=rms >= self._activity_threshold,
        )
        self._last_activity_candidate = candidate
        self._enqueue("activity", candidate)

    def _enqueue(self, name: str, item: Any) -> None:
        queue = self._queues.get(name)
        if queue is None:
            return
        self._ensure_worker(name)
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            self.metrics.consumer_overflow_frames[name] += 1

    def _ensure_worker(self, name: str) -> None:
        if name not in self._workers:
            self._workers[name] = asyncio.create_task(
                self._run_consumer(name), name=f"audio-ingress-{name}"
            )

    async def _run_consumer(self, name: str) -> None:
        queue = self._queues[name]
        consumer = self._consumers[name]
        while True:
            item = await queue.get()
            try:
                if item is self._STOP:
                    return
                if self._native_async_consumers[name]:
                    result = consumer(item)
                else:
                    result = await asyncio.to_thread(consumer, item)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                self._worker_errors.setdefault(name, error)
            finally:
                queue.task_done()

    def _raise_worker_error(self) -> None:
        if not self._worker_errors:
            return
        name, error = next(iter(self._worker_errors.items()))
        raise AudioIngressConsumerError(f"{name} consumer failed: {error}") from error


def _pcm16_rms(pcm: bytes) -> float:
    """Return root-mean-square amplitude for little-endian signed PCM16."""
    samples = struct.iter_unpack("<h", pcm)
    total = 0
    count = 0
    for (sample,) in samples:
        total += sample * sample
        count += 1
    return math.sqrt(total / count) if count else 0.0


def _is_native_async_callable(
    consumer: Callable[[Any], Awaitable[None] | None],
) -> bool:
    """Return whether invoking ``consumer`` itself must stay on this event loop."""
    return inspect.iscoroutinefunction(consumer) or inspect.iscoroutinefunction(
        getattr(consumer, "__call__", None)
    )
