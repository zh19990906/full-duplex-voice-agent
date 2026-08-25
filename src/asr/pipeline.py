"""Model-independent streaming ASR pipeline."""

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from typing import Any

from src.adapters.asr.base import BaseASRAdapter
from src.audio.frames import AudioFrame
from src.core.events.events import BaseEvent, UserSpeechPartialEvent, UserTurnEndEvent

from .session import ASRSession, ASRSessionStatus
from .stream import TranscriptChunk


EventHandler = Callable[[BaseEvent], Awaitable[None] | None]


class ASRPipeline:
    """Connect audio input to an ASR adapter and stable realtime events.

    The adapter remains the only speech-understanding implementation. This
    pipeline only normalizes generic adapter results and emits project events.
    """

    def __init__(
        self,
        adapter: BaseASRAdapter,
        event_handler: EventHandler | None = None,
    ) -> None:
        self.adapter = adapter
        self.event_handler = event_handler
        self.session: ASRSession | None = None
        self.emitted_events: list[BaseEvent] = []
        self._chunk_number = 0
        self._stream_task: asyncio.Task[Any] | None = None

    async def start_session(self, session_id: str, language: str) -> ASRSession:
        """Create and activate a streaming ASR session."""
        if self.session is not None and self.session.status is ASRSessionStatus.RUNNING:
            await self.stop_session()
        reset = getattr(self.adapter, "reset", None)
        if callable(reset):
            reset()
        self.session = ASRSession(
            session_id=session_id,
            language=language,
            status=ASRSessionStatus.RUNNING,
        )
        self._chunk_number = 0
        return self.session

    async def push_audio(self, audio: bytes | AudioFrame) -> tuple[BaseEvent, ...]:
        """Send audio to the adapter and convert returned chunks to events."""
        session = self._require_running_session()
        audio_chunk = audio.data if isinstance(audio, AudioFrame) else audio
        events: list[BaseEvent] = []
        current_task = asyncio.current_task()
        self._stream_task = current_task
        try:
            result = await self.adapter.stream_audio(audio_chunk)
            async for item in self._iter_result(result):
                if not self._is_current_running_session(session):
                    break
                for chunk in self._normalize_result(item):
                    if not self._is_current_running_session(session):
                        break
                    event = self._event_for_chunk(chunk)
                    self.emitted_events.append(event)
                    events.append(event)
                    if self.event_handler is not None:
                        handled = self.event_handler(event)
                        if inspect.isawaitable(handled):
                            await handled
        except asyncio.CancelledError:
            # Cancellation is a normal interruption boundary; already emitted
            # events remain observable, while no late provider result escapes.
            return tuple(events)
        finally:
            if self._stream_task is current_task:
                self._stream_task = None
        return tuple(events)

    async def stop_session(self) -> ASRSession:
        """Stop accepting audio for the current session."""
        session = self._require_session()
        session.status = ASRSessionStatus.STOPPED
        stream_task = self._stream_task
        if stream_task is not None and stream_task is not asyncio.current_task():
            stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True)
        cancel = getattr(self.adapter, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result
        return session

    async def complete_session(self) -> ASRSession:
        """Mark the current session completed without inference logic."""
        session = self._require_session()
        session.status = ASRSessionStatus.COMPLETED
        return session

    def _require_session(self) -> ASRSession:
        if self.session is None:
            raise RuntimeError("ASR session has not been started")
        return self.session

    def _require_running_session(self) -> ASRSession:
        session = self._require_session()
        if session.status is not ASRSessionStatus.RUNNING:
            raise RuntimeError("ASR session is not running")
        return session

    def _is_current_running_session(self, session: ASRSession) -> bool:
        return self.session is session and session.status is ASRSessionStatus.RUNNING

    def _event_for_chunk(self, chunk: TranscriptChunk) -> BaseEvent:
        session = self._require_running_session()
        self._chunk_number += 1
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "is_final": chunk.is_final,
            "session_id": session.session_id,
            "language": session.language,
            "revision_id": chunk.revision_id,
            "unstable_text": chunk.unstable_text,
            "committed_text": chunk.committed_text,
            "replaces_committed": chunk.replaces_committed,
        }
        event_type = UserTurnEndEvent if chunk.is_final else UserSpeechPartialEvent
        return event_type(
            event_id=f"asr-{session.session_id}-{self._chunk_number}",
            timestamp=chunk.timestamp,
            source="asr",
            payload=payload,
        )

    @classmethod
    def _normalize_result(cls, result: Any) -> tuple[TranscriptChunk, ...]:
        if result is None:
            return ()
        if isinstance(result, (list, tuple)):
            chunks: list[TranscriptChunk] = []
            for item in result:
                chunks.extend(cls._normalize_result(item))
            return tuple(chunks)
        if isinstance(result, TranscriptChunk):
            return (result,)
        if isinstance(result, str):
            return (TranscriptChunk("adapter-chunk", result, time.time(), False),)
        if isinstance(result, Mapping):
            text = result.get("text", "")
            if not isinstance(text, str):
                raise TypeError("ASR result text must be a string")
            return (
                TranscriptChunk(
                    chunk_id=str(result.get("chunk_id", "adapter-chunk")),
                    text=text,
                    timestamp=float(result.get("timestamp", time.time())),
                    is_final=bool(result.get("is_final", False)),
                    revision_id=result.get("revision_id", 0),
                    unstable_text=result.get("unstable_text", ""),
                    committed_text=result.get("committed_text"),
                    replaces_committed=result.get("replaces_committed", False),
                ),
            )
        raise TypeError("ASR adapter result must be text, mapping, chunk, or None")

    @staticmethod
    async def _iter_result(result: Any) -> AsyncIterable[Any]:
        """Yield normal or async adapter output one item at a time."""
        if result is None:
            return
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
            return
        if isinstance(result, (list, tuple)):
            for item in result:
                yield item
            return
        yield result


StreamingASRPipeline = ASRPipeline
