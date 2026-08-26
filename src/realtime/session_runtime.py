"""Minimal per-session composition root for the realtime audio path."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from src.asr.stream import TranscriptChunk
from src.controller.actions import ActionType, ControllerAction
from src.core.events.events import BaseEvent, UserSpeechPartialEvent, UserTurnEndEvent
from src.realtime.interpretation import (
    InterpretationPipeline,
    InterpretationSession,
    TranslationSegment,
)
from src.realtime.session_state import ConversationMode
from src.realtime.text_segmenter import TextSegment
from src.tts_runtime.stream import AudioChunk

from .audio_ingress import AudioIngress
from .cancellation import CancellationToken
from .checkpoint import ResponseCheckpointStore, ResumePlan
from .identifiers import GenerationClock, IdentifierAllocator
from .playback import PlaybackAck, PlaybackCoordinator


CancelGenerationHook = Any


@dataclass
class RuntimeActionEffect:
    """Observable runtime side effects from controller actions."""

    archived_response_id: str | None = None
    advanced_epoch: int | None = None
    resume_plan: ResumePlan | None = None


class RealtimeSessionRuntime:
    """Own session identity, epoch fencing, checkpoints, and audio ingress lifecycle."""

    def __init__(
        self,
        session_id: str,
        *,
        ingress: AudioIngress | None = None,
        playback: PlaybackCoordinator | None = None,
        checkpoints: ResponseCheckpointStore | None = None,
        cancel_generation: CancelGenerationHook | None = None,
        interpretation_translator: Any | None = None,
        interpretation_sink: Any | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = session_id
        self.generation_clock = GenerationClock()
        self._interpretation_response_ids = IdentifierAllocator()
        self.ingress = ingress if ingress is not None else AudioIngress()
        self.playback = playback if playback is not None else PlaybackCoordinator()
        self.checkpoints = checkpoints if checkpoints is not None else self.playback.checkpoint_store
        self.playback.set_checkpoint_store(self.checkpoints)
        self.cancel_generation = cancel_generation
        self.interpretation_translator = interpretation_translator
        self.interpretation_sink = interpretation_sink
        self.conversation_mode = ConversationMode.CHAT
        self.interpretation_session: InterpretationSession | None = None
        self._interpretation_pipeline: InterpretationPipeline | None = None
        self._interpretation_cancellation: CancellationToken | None = None
        self.current_response_id: str | None = None
        self._archived_response_ids: list[str] = []
        self._closed = False

    @property
    def generation_epoch(self) -> int:
        """Return the active generation epoch for this session."""
        return self.generation_clock.current

    @property
    def archived_response_ids(self) -> tuple[str, ...]:
        return tuple(self._archived_response_ids)

    @property
    def closed(self) -> bool:
        """Return whether the session runtime has been closed."""
        return self._closed

    def activate_response(self, response_id: str) -> None:
        self.current_response_id = response_id
        self.checkpoints.activate(response_id, generation_epoch=self.generation_epoch)
        self.playback.set_active_response(response_id, generation_epoch=self.generation_epoch)

    def record_generated_text(self, response_id: str, text: str) -> None:
        if response_id != self.current_response_id:
            return
        self.checkpoints.record_generated_text(response_id, text)

    def record_response_segment(self, segment: TextSegment) -> bool:
        if segment.response_id != self.current_response_id:
            return False
        if segment.generation_epoch != self.generation_epoch:
            return False
        self.checkpoints.record_segment(segment.response_id, segment.segment_id, segment.text)
        return True

    def record_audio_chunk(self, chunk: AudioChunk) -> bool:
        if chunk.response_id != self.current_response_id:
            return False
        if chunk.generation_epoch != self.generation_epoch:
            return False
        if chunk.segment_id is None:
            return False
        self.checkpoints.record_segment(
            chunk.response_id,
            chunk.segment_id,
            "",
            audio=chunk.audio_data,
        )
        return True

    def ack_playback(
        self,
        response_id: str,
        *,
        generation_epoch: int,
        segment_id: int,
        sample_offset: int,
        audio_time: float = 0.0,
        playback_attempt_id: int | None = None,
    ) -> bool:
        return self.playback.ack(
            PlaybackAck(
                response_id=response_id,
                generation_epoch=generation_epoch,
                segment_id=segment_id,
                sample_offset=sample_offset,
                audio_time=audio_time,
                playback_attempt_id=playback_attempt_id,
            )
        )

    def advance_generation(self) -> int:
        """Invalidate current response work and return the replacement epoch."""
        new_epoch = self.generation_clock.advance()
        self._restart_interpretation_pipeline()
        if self.current_response_id is not None:
            self.playback.set_active_response(
                self.current_response_id,
                generation_epoch=new_epoch,
            )
        return new_epoch

    async def apply_controller_actions(
        self,
        actions: tuple[ControllerAction, ...],
    ) -> RuntimeActionEffect:
        effect = RuntimeActionEffect()
        revise_requested = False
        for action in actions:
            if action.action_type in {
                ActionType.DUCK_RESPONSE,
                ActionType.RESTORE_RESPONSE,
                ActionType.PAUSE_RESPONSE,
                ActionType.RESUME_RESPONSE,
                ActionType.STOP_RESPONSE,
            }:
                resume_plan = await self.playback.apply(action)
                if resume_plan is not None:
                    effect.resume_plan = resume_plan
                continue
            if action.action_type is ActionType.REVISE_RESPONSE:
                effect.advanced_epoch = self._advance_for_current_response(archive=False)
                revise_requested = True
                continue
            if action.action_type is ActionType.CANCEL_GENERATION:
                await self._cancel_current_generation()
                continue
            if action.action_type is ActionType.SWITCH_MODE:
                self._apply_mode_switch(action)
                continue
            if action.action_type is ActionType.PROCESS_USER_REQUEST:
                if not revise_requested and self.current_response_id is not None:
                    effect.archived_response_id = self.current_response_id
                    effect.advanced_epoch = self._advance_for_current_response(archive=True)
                continue
        return effect

    async def push_audio(self, frame: bytes) -> None:
        """Delegate a browser PCM frame to this session's ingress."""
        if self._closed:
            raise RuntimeError("realtime session runtime is closed")
        await self.ingress.push(frame)

    async def flush(self) -> None:
        """Wait for this session's accepted audio to reach all consumers."""
        await self.ingress.flush()

    async def close(self) -> None:
        """Close the ingress and make the session unavailable for more audio."""
        if self._closed:
            await self.ingress.close()
            return
        self._closed = True
        self._clear_interpretation_pipeline(clear_session=True)
        await self.ingress.close()

    async def accept_asr_event(self, event: BaseEvent) -> TranslationSegment | None:
        """Consume ASR pipeline events through the live interpretation path."""
        if not isinstance(event, BaseEvent):
            raise TypeError("event must be BaseEvent")
        if not isinstance(event, (UserSpeechPartialEvent, UserTurnEndEvent)):
            return None
        return await self.accept_transcript_chunk(self._chunk_from_event(event))

    async def accept_transcript_chunk(
        self,
        chunk: TranscriptChunk,
    ) -> TranslationSegment | None:
        """Consume one revision-aware transcript chunk in interpretation mode."""
        if not isinstance(chunk, TranscriptChunk):
            raise TypeError("chunk must be TranscriptChunk")
        if self.conversation_mode is not ConversationMode.INTERPRETATION:
            return None
        pipeline = self._interpretation_pipeline
        if pipeline is None:
            return None
        return await pipeline.push_chunk(chunk)

    def _advance_for_current_response(self, *, archive: bool) -> int:
        response_id = self.current_response_id
        new_epoch = self.advance_generation()
        if response_id is None:
            return new_epoch
        if archive:
            self.checkpoints.archive(response_id)
            self._archived_response_ids.append(response_id)
        else:
            self.checkpoints.discard_unplayed(response_id)
        return new_epoch

    async def _cancel_current_generation(self) -> None:
        if self.cancel_generation is None:
            return
        result = self.cancel_generation(self.current_response_id, self.generation_epoch)
        if inspect.isawaitable(result):
            await result

    def _apply_mode_switch(self, action: ControllerAction) -> None:
        payload = dict(action.payload or {})
        target_mode = payload.get("target_mode")
        if target_mode == ConversationMode.CHAT.value:
            self.conversation_mode = ConversationMode.CHAT
            self._clear_interpretation_pipeline(clear_session=True)
            return
        if target_mode != ConversationMode.INTERPRETATION.value:
            raise ValueError("unsupported target_mode")
        target_language = payload.get("target_language")
        source_language = payload.get("source_language")
        if self.interpretation_session is None:
            self._build_interpretation_pipeline(
                target_language=target_language,
                source_language=source_language,
            )
        else:
            self.interpretation_session.set_target_language(
                target_language,
                source_language=source_language,
            )
            if self._interpretation_pipeline is not None:
                self._interpretation_pipeline.set_target_language(
                    target_language,
                    source_language=source_language,
                )
        self.conversation_mode = ConversationMode.INTERPRETATION

    def _build_interpretation_pipeline(
        self,
        *,
        target_language: str,
        source_language: str | None,
    ) -> None:
        if self.interpretation_translator is None or self.interpretation_sink is None:
            self.interpretation_session = InterpretationSession(
                target_language=target_language,
                source_language=source_language,
                generation_clock=self.generation_clock,
                response_id_factory=self._interpretation_response_ids.next_response_id,
            )
            self._interpretation_pipeline = None
            self._interpretation_cancellation = None
            self.activate_response(self.interpretation_session.response_id)
            return
        cancellation = CancellationToken()
        pipeline = InterpretationPipeline(
            translator=self.interpretation_translator,
            sink=self.interpretation_sink,
            target_language=target_language,
            source_language=source_language,
            cancellation_token=cancellation,
            generation_clock=self.generation_clock,
            response_id_factory=self._interpretation_response_ids.next_response_id,
        )
        self._interpretation_cancellation = cancellation
        self._interpretation_pipeline = pipeline
        self.interpretation_session = pipeline.session
        self.activate_response(pipeline.session.response_id)

    def _restart_interpretation_pipeline(self) -> None:
        if self.conversation_mode is not ConversationMode.INTERPRETATION:
            return
        session = self.interpretation_session
        if session is None:
            return
        target_language = session.target_language
        source_language = session.source_language
        self._clear_interpretation_pipeline(clear_session=False)
        self._build_interpretation_pipeline(
            target_language=target_language,
            source_language=source_language,
        )

    def _clear_interpretation_pipeline(self, *, clear_session: bool) -> None:
        if self._interpretation_cancellation is not None:
            self._interpretation_cancellation.cancel()
        self._interpretation_pipeline = None
        self._interpretation_cancellation = None
        if clear_session:
            self.interpretation_session = None

    @staticmethod
    def _chunk_from_event(event: BaseEvent) -> TranscriptChunk:
        payload = event.payload
        return TranscriptChunk(
            chunk_id=str(payload.get("chunk_id", event.event_id)),
            text=str(payload.get("text", "")),
            timestamp=float(event.timestamp),
            is_final=isinstance(event, UserTurnEndEvent),
            revision_id=int(payload.get("revision_id", 0)),
            unstable_text=str(payload.get("unstable_text", "")),
            committed_text=payload.get("committed_text"),
            replaces_committed=bool(payload.get("replaces_committed", False)),
        )
