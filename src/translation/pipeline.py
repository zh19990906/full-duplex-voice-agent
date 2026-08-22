"""Model-independent streaming translation orchestration."""

import time
from collections.abc import Mapping
from typing import Any

from src.core.interfaces.asr import ASRAdapter
from src.core.interfaces.translation import TranslationAdapter
from src.core.interfaces.tts import TTSAdapter

from .session import TranslationSession, TranslationSessionStatus
from .stream import TranslationChunk


class StreamingTranslationPipeline:
    """Connect translated text chunks to the translation and TTS adapters.

    ``push_text`` is the boundary for text emitted by a streaming ASR layer.
    The current ASR contract accepts audio but does not expose transcript
    values, so this pipeline does not invoke ASR itself.
    """

    def __init__(
        self,
        translation_adapter: TranslationAdapter,
        tts_adapter: TTSAdapter,
        asr_adapter: ASRAdapter | None = None,
    ) -> None:
        self.translation_adapter = translation_adapter
        self.tts_adapter = tts_adapter
        self.asr_adapter = asr_adapter
        self._session: TranslationSession | None = None
        self._chunk_number = 0

    @property
    def session(self) -> TranslationSession | None:
        """Return the current session, if one has been started."""
        return self._session

    async def start_session(
        self,
        session_id: str,
        source_language: str,
        target_language: str,
    ) -> TranslationSession:
        """Create and start a translation session."""
        self._session = TranslationSession(
            session_id=session_id,
            source_language=source_language,
            target_language=target_language,
            status=TranslationSessionStatus.RUNNING,
        )
        self._chunk_number = 0
        return self._session

    async def push_text(self, text: str, is_final: bool = False) -> TranslationChunk:
        """Translate one ASR text update and forward output to streaming TTS."""
        self._require_running_session()
        result = await self.translation_adapter.translate_stream(text)
        translated_text = self._translated_text(result)
        chunk = TranslationChunk(
            chunk_id=f"{self._session.session_id}:{self._chunk_number}",
            source_text=text,
            translated_text=translated_text,
            timestamp=time.time(),
            is_final=is_final,
        )
        self._chunk_number += 1
        if translated_text:
            await self.tts_adapter.synthesize(translated_text)
            await self.tts_adapter.stream_audio(translated_text)
        return chunk

    async def stop_session(self) -> TranslationSession:
        """Mark the current session stopped without executing model logic."""
        session = self._require_session()
        session.status = TranslationSessionStatus.STOPPED
        return session

    async def complete_session(self) -> TranslationSession:
        """Mark the current session completed."""
        session = self._require_session()
        session.status = TranslationSessionStatus.COMPLETED
        return session

    def _require_session(self) -> TranslationSession:
        if self._session is None:
            raise RuntimeError("translation session has not been started")
        return self._session

    def _require_running_session(self) -> TranslationSession:
        session = self._require_session()
        if session.status is not TranslationSessionStatus.RUNNING:
            raise RuntimeError("translation session is not running")
        return session

    @staticmethod
    def _translated_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, TranslationChunk):
            return result.translated_text
        if isinstance(result, Mapping):
            value = result.get("translated_text", "")
            if not isinstance(value, str):
                raise TypeError("translation result translated_text must be a string")
            return value
        raise TypeError("translation adapter result must be text, mapping, chunk, or None")
