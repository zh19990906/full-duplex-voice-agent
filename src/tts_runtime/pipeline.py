"""Model-independent streaming TTS pipeline."""

import inspect
import time
from collections.abc import AsyncIterable, Mapping
from typing import Any

from src.adapters.tts.base import BaseTTSAdapter
from src.audio.frames import AudioFrame
from src.llm_runtime.stream import TokenChunk

from .session import TTSSession, TTSSessionStatus
from .stream import AudioChunk


class TTSRuntimePipeline:
    """Convert streaming text into audio chunks and transport frames."""

    def __init__(
        self,
        adapter: BaseTTSAdapter,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.adapter = adapter
        self.sample_rate = sample_rate
        self.channels = channels
        self.session: TTSSession | None = None
        self._audio_queue: list[AudioFrame] = []
        self._chunk_number = 0

    async def start_session(self, session_id: str) -> TTSSession:
        """Create a running TTS session and reset adapter state."""
        if self.session is not None and self.session.status is TTSSessionStatus.RUNNING:
            await self.interrupt()
        reset = getattr(self.adapter, "reset", None)
        if callable(reset):
            reset()
        self.session = TTSSession(
            session_id=session_id,
            status=TTSSessionStatus.RUNNING,
            started_at=time.time(),
        )
        self._audio_queue.clear()
        self._chunk_number = 0
        return self.session

    async def push_text(self, text: str | TokenChunk) -> tuple[AudioChunk, ...]:
        """Synthesize one text/token update and queue resulting audio frames."""
        session = self._require_running_session()
        source_text = text.text if isinstance(text, TokenChunk) else text
        await self.adapter.synthesize(source_text)
        result = self.adapter.stream_audio(source_text)
        if inspect.isawaitable(result):
            result = await result

        chunks: list[AudioChunk] = []
        async for item in self._iter_result(result):
            if not self._is_current_running_session(session):
                break
            for chunk in self._normalize_result(item):
                if not self._is_current_running_session(session):
                    break
                chunks.append(chunk)
                self._audio_queue.append(self._to_audio_frame(chunk))
        return tuple(chunks)

    async def stream_audio(self) -> tuple[AudioFrame, ...]:
        """Drain queued audio frames for the transport layer."""
        frames = tuple(self._audio_queue)
        self._audio_queue.clear()
        return frames

    async def interrupt(self) -> TTSSession:
        """Flush queued audio, interrupt the adapter, and block stale output."""
        session = self._require_session()
        session.status = TTSSessionStatus.INTERRUPTED
        session.interrupted_at = session.interrupted_at or time.time()
        self._audio_queue.clear()
        await self.adapter.interrupt()
        return session

    async def complete_session(self) -> TTSSession:
        """Complete a session unless it has already been interrupted."""
        session = self._require_session()
        if session.status is TTSSessionStatus.RUNNING:
            session.status = TTSSessionStatus.COMPLETED
            session.completed_at = time.time()
        return session

    def _require_session(self) -> TTSSession:
        if self.session is None:
            raise RuntimeError("TTS session has not been started")
        return self.session

    def _require_running_session(self) -> TTSSession:
        session = self._require_session()
        if session.status is not TTSSessionStatus.RUNNING:
            raise RuntimeError("TTS session is not running")
        return session

    def _is_current_running_session(self, session: TTSSession) -> bool:
        return self.session is session and session.status is TTSSessionStatus.RUNNING

    async def _iter_result(self, result: Any) -> AsyncIterable[Any]:
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

    def _normalize_result(self, result: Any) -> tuple[AudioChunk, ...]:
        if isinstance(result, AudioChunk):
            return (result,)
        if isinstance(result, AudioFrame):
            return (
                AudioChunk(result.frame_id, result.data, result.timestamp, False),
            )
        if isinstance(result, bytes):
            return (
                AudioChunk(
                    f"{self.session.session_id}:{self._chunk_number}",
                    result,
                    time.time(),
                    False,
                ),
            )
        if isinstance(result, Mapping):
            audio_data = result.get("audio_data", b"")
            if not isinstance(audio_data, bytes):
                raise TypeError("TTS result audio_data must be bytes")
            return (
                AudioChunk(
                    str(result.get("chunk_id", f"{self.session.session_id}:{self._chunk_number}")),
                    audio_data,
                    float(result.get("timestamp", time.time())),
                    bool(result.get("is_final", False)),
                ),
            )
        raise TypeError("TTS adapter result must be bytes, mapping, AudioChunk, or AudioFrame")

    def _to_audio_frame(self, chunk: AudioChunk) -> AudioFrame:
        frame = AudioFrame(
            frame_id=chunk.chunk_id,
            timestamp=chunk.timestamp,
            sample_rate=self.sample_rate,
            channels=self.channels,
            data=chunk.audio_data,
        )
        self._chunk_number += 1
        return frame


StreamingTTSPipeline = TTSRuntimePipeline
TTSPipeline = TTSRuntimePipeline
