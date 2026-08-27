"""Dependency containers for legacy application and realtime runtime composition."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from src.application.voice_agent import VoiceAgent
from src.asr.stream import TranscriptChunk
from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.llm_runtime.stream import TokenChunk
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.cancellation import ActiveTaskSlot, CancellationToken
from src.realtime.playback import PlaybackCoordinator
from src.realtime.policy import PolicyRequest, SemanticPolicyEngine
from src.realtime.response_pipeline import RealtimeResponsePipeline, ResponseStreamEnd
from src.realtime.interpretation import TranslationSegment
from src.realtime.supervisor import WorkerSupervisor, WorkerTerminalEvent
from src.realtime.session_runtime import RealtimeSessionRuntime
from src.realtime.session_state import ConversationMode
from src.realtime.session_state import FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent, SpeechEventFusion
from src.realtime.text_segmenter import LanguageAwareTextSegmenter, TextSegment
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler
from src.tts_runtime.stream import AudioChunk
from src.adapters.turn.config import X2TurnConfig
from src.model_runtime.factory import (
    create_llm_adapter,
    create_policy_provider,
    create_realtime_asr_provider,
    create_tts_adapter,
    create_turn_provider,
)
from src.model_runtime.manager import ModelManager


class ApplicationContainer:
    """Create and wire the model-independent application components.

    Dependencies are accepted as optional constructor arguments so tests and
    future deployments can replace infrastructure without changing the
    composition boundary. Model adapters are intentionally not constructed.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        controller: ConversationController | None = None,
        generation_manager: GenerationManager | None = None,
        scheduler: RealtimeScheduler | None = None,
    ) -> None:
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.controller = controller if controller is not None else ConversationController()
        self.generation_manager = (
            generation_manager if generation_manager is not None else GenerationManager()
        )
        self.scheduler = scheduler if scheduler is not None else RealtimeScheduler()
        self.voice_agent = VoiceAgent(
            event_bus=self.event_bus,
            controller=self.controller,
            generation_manager=self.generation_manager,
        )
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Whether application event subscriptions are active."""
        return self._initialized

    async def initialize(self) -> None:
        """Start application-level event subscriptions."""
        if self._initialized:
            return
        await self.voice_agent.start()
        self._initialized = True

    async def shutdown(self) -> None:
        """Stop application subscriptions and the scheduler."""
        await self.voice_agent.stop()
        await self.scheduler.shutdown()
        self._initialized = False


_EVENT_STREAM_STOP = object()


class SlowConsumerError(RuntimeError):
    """Raised when a bounded outbound event queue cannot keep up."""


class QwenTranslationAdapter:
    """Adapt a Qwen-compatible local runtime to the translation contract."""

    def __init__(
        self,
        model_path: str,
        *,
        runtime: Any,
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("translation model_path must be supplied")
        if runtime is None:
            raise ValueError("translation runtime must be supplied")
        self.model_path = model_path
        self.runtime = runtime
        self.device = device
        self.options = dict(options or {})

    async def translate_stream(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        method = getattr(self.runtime, "translate_stream", None)
        if callable(method):
            result = method(prompt, **self.options)
        else:
            method = getattr(self.runtime, "generate", None)
            if not callable(method):
                raise RuntimeError("translation runtime must expose translate_stream() or generate()")
            result = method(prompt, **self.options)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            parts = []
            async for item in result:
                parts.append(_coerce_text(item))
            return "".join(parts)
        if isinstance(result, (list, tuple)):
            return "".join(_coerce_text(item) for item in result)
        return _coerce_text(result)


class RuntimeBackedTranslationSink:
    """Route accepted interpretation translations through runtime playback/events."""

    def __init__(self, runtime: "ServerRealtimeSessionRuntime") -> None:
        self.runtime = runtime

    async def publish(self, segment: TranslationSegment) -> None:
        if self.runtime.closed:
            return
        segment.playback_started = True
        text_segment = TextSegment(
            segment.translation_segment_id,
            segment.translated_text,
            False,
            segment.response_id,
            segment.generation_epoch,
        )
        self.runtime.record_response_segment(text_segment)
        self.runtime.session_state.response = ResponseState.PLAYING
        self.runtime.session_state.floor = FloorState.ASSISTANT
        await self.runtime._publish(
            {
                "event": "translation",
                "response_id": segment.response_id,
                "generation_epoch": segment.generation_epoch,
                "segment_id": segment.translation_segment_id,
                "payload": asdict(segment),
            }
        )
        request_id = self.runtime._tts_request_id(text_segment)
        stream = await self.runtime._start_tts_stream(text_segment, request_id)
        async for raw_chunk in _iterate_maybe_async(stream):
            chunk = self.runtime._tag_audio(
                raw_chunk,
                text_segment,
                request_id,
                self.runtime.playback.active_playback_attempt_id,
            )
            self.runtime.record_audio_chunk(chunk)
            await self.runtime._publish_audio_chunk(chunk)


class _SharedRuntimeBoundary:
    """Serialize access to one shared heavyweight runtime object."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._operation_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._owner_lock = asyncio.Lock()
        self._active_owner: object | None = None

    async def call(self, owner: object, method_name: str, *args: Any, **kwargs: Any) -> Any:
        async with self._operation_lock:
            await self._set_active_owner(owner)
            try:
                method = getattr(self.runtime, method_name, None)
                if not callable(method):
                    raise RuntimeError(f"shared runtime must expose {method_name}()")
                result = method(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
            finally:
                await self._clear_active_owner(owner)

    async def iterate(
        self,
        owner: object,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async with self._operation_lock:
            await self._set_active_owner(owner)
            try:
                method = getattr(self.runtime, method_name, None)
                if not callable(method):
                    raise RuntimeError(f"shared runtime must expose {method_name}()")
                result = method(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                async for item in _iterate_maybe_async(result):
                    yield item
            finally:
                await self._clear_active_owner(owner)

    async def control(
        self,
        owner: object,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        async with self._control_lock:
            active_owner = await self._get_active_owner()
            if active_owner is owner:
                return await self._invoke_control(method_name, *args, **kwargs)
            if active_owner is not None:
                return False
            async with self._operation_lock:
                active_owner = await self._get_active_owner()
                if active_owner is owner:
                    return await self._invoke_control(method_name, *args, **kwargs)
                if active_owner is not None:
                    return False
                return await self._invoke_control(method_name, *args, **kwargs)

    async def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def _get_active_owner(self) -> object | None:
        async with self._owner_lock:
            return self._active_owner

    async def _set_active_owner(self, owner: object) -> None:
        async with self._owner_lock:
            self._active_owner = owner

    async def _clear_active_owner(self, owner: object) -> None:
        async with self._owner_lock:
            if self._active_owner is owner:
                self._active_owner = None

    async def _invoke_control(self, method_name: str, *args: Any, **kwargs: Any) -> bool:
        method = getattr(self.runtime, method_name, None)
        if not callable(method):
            return False
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
        return True


class _SessionRuntimeProxy:
    """Per-session façade around one serialized heavyweight runtime boundary."""

    def __init__(self, boundary: _SharedRuntimeBoundary) -> None:
        self.runtime = boundary

    async def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        return await self.runtime.call(self, "transcribe", *args, **kwargs)

    async def infer(self, *args: Any, **kwargs: Any) -> Any:
        return await self.runtime.call(self, "infer", *args, **kwargs)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        return await self.runtime.call(self, "generate", *args, **kwargs)

    async def translate_stream(self, *args: Any, **kwargs: Any) -> Any:
        method_name = "translate_stream" if callable(getattr(self.runtime.runtime, "translate_stream", None)) else "generate"
        return await self.runtime.call(self, method_name, *args, **kwargs)

    async def stream_tokens(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for item in self.runtime.iterate(self, "stream_tokens", *args, **kwargs):
            yield item

    async def stream_audio(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for item in self.runtime.iterate(self, "stream_audio", *args, **kwargs):
            yield item

    async def cancel(self, *args: Any, **kwargs: Any) -> bool:
        return await self.runtime.control(self, "cancel", *args, **kwargs)

    async def interrupt(self, *args: Any, **kwargs: Any) -> bool:
        return await self.runtime.control(self, "interrupt", *args, **kwargs)

    def reset(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.runtime.control(self, "reset"))
            return None
        return loop.create_task(self.runtime.control(self, "reset"))


class ServerRealtimeSessionRuntime(RealtimeSessionRuntime):
    """Per-session realtime runtime that owns command, audio, and event flow."""

    def __init__(
        self,
        session_id: str,
        *,
        llm: Any,
        tts: Any,
        asr: Any | None = None,
        turn: Any | None = None,
        policy_engine: SemanticPolicyEngine | None = None,
        interpretation_translator: Any | None = None,
        interpretation_sink: Any | None = None,
        controller: ConversationController | None = None,
        fusion: SpeechEventFusion | None = None,
        prompt_builder: Callable[[str], str] | None = None,
        segmenter: LanguageAwareTextSegmenter | None = None,
        tts_queue_capacity: int = 2,
        outbound_event_queue_capacity: int = 64,
        audio_output_sample_rate: int = 24000,
        audio_output_channels: int = 1,
    ) -> None:
        if outbound_event_queue_capacity < 1:
            raise ValueError("outbound_event_queue_capacity must be at least one")
        self._events: asyncio.Queue[Any] = asyncio.Queue(maxsize=outbound_event_queue_capacity)
        self.llm = llm
        self.tts = tts
        self.asr = asr
        self.turn = turn
        self.policy_engine = policy_engine
        self.interpretation_translator = interpretation_translator
        self.controller = controller if controller is not None else ConversationController()
        self.prompt_builder = prompt_builder or _default_prompt_builder
        self.segmenter = segmenter or LanguageAwareTextSegmenter()
        self.tts_queue_capacity = tts_queue_capacity
        self.audio_output_sample_rate = audio_output_sample_rate
        self.audio_output_channels = audio_output_channels
        self.session_state = SessionState()
        self.fusion = fusion if fusion is not None else SpeechEventFusion(session_state=self.session_state)
        self._response_identifiers = 0
        self._connected = 0
        self._started = False
        self._closed_stream = False
        self._terminal_error: BaseException | None = None
        self._generation_slot = ActiveTaskSlot()
        self._generation_token: CancellationToken | None = None
        self._assistant_last_text = ""
        self._playback_failed = False
        self._sender_playback = PlaybackCoordinator(send_command=self._publish_command)
        self._tts_supervisor = WorkerSupervisor("tts_worker", max_restarts=1)
        self._runtime_interpretation_sink = interpretation_sink or RuntimeBackedTranslationSink(self)
        super().__init__(
            session_id,
            playback=self._sender_playback,
            cancel_generation=self._cancel_generation_hook,
            interpretation_translator=interpretation_translator,
            interpretation_sink=self._runtime_interpretation_sink if interpretation_translator is not None else interpretation_sink,
        )

    async def start(self) -> None:
        self._started = True

    async def connect(self) -> None:
        self._connected += 1
        await self._publish({"event": "session_snapshot", "payload": self.snapshot()})

    async def disconnect(self) -> None:
        self._connected = max(0, self._connected - 1)

    async def events(self) -> AsyncIterator[Any]:
        while True:
            if self._closed_stream and self._events.empty():
                error = self._terminal_error
                if error is not None:
                    raise error
                return
            item = await self._events.get()
            if item is _EVENT_STREAM_STOP:
                error = self._terminal_error
                if error is not None:
                    raise error
                return
            yield item
            if self._closed_stream and self._events.empty():
                error = self._terminal_error
                if error is not None:
                    raise error
                return

    async def accept_audio_frame(self, frame: RealtimeAudioFrame) -> None:
        await super().accept_audio_frame(frame)
        await self._publish(
            {
                "event": "AUDIO_FRAME_ACCEPTED",
                "response_id": self.current_response_id,
                "generation_epoch": self.generation_epoch,
                "payload": {"sequence": frame.header.sequence},
            }
        )
        await self._process_audio_frame(frame)

    async def accept_command(self, command: Mapping[str, Any]) -> None:
        if not isinstance(command, Mapping):
            raise TypeError("command must be a mapping")
        command_type = str(command.get("type", "")).strip().lower()
        payload = command.get("payload")
        if payload is None:
            payload = command
        if command_type in {"text", "message"}:
            text = command.get("text", command.get("message"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text command requires non-empty text")
            await self._start_generation(text.strip())
            return
        if command_type == "playback_ack":
            self._handle_playback_ack(payload)
            return
        if command_type == "resume_response":
            self.session_state.response = ResponseState.PLAYING
            self.session_state.floor = FloorState.ASSISTANT
            await self.apply_controller_actions((ControllerAction(ActionType.RESUME_RESPONSE),))
            return
        if command_type == "pause_response":
            self.session_state.response = ResponseState.PAUSED
            self.session_state.floor = FloorState.USER
            await self.apply_controller_actions((ControllerAction(ActionType.PAUSE_RESPONSE),))
            return
        if command_type == "interrupt":
            self.session_state.response = ResponseState.CANCELLING
            self.session_state.floor = FloorState.USER
            await self.apply_controller_actions(
                (
                    ControllerAction(ActionType.STOP_RESPONSE),
                    ControllerAction(ActionType.CANCEL_GENERATION),
                )
            )
            return
        raise ValueError(f"unsupported command type: {command_type or '<empty>'}")

    async def close(self) -> None:
        if self._closed_stream:
            return
        await self._cancel_generation()
        await self._generation_slot.cancel()
        await super().close()
        self._closed_stream = True
        try:
            self._events.put_nowait(_EVENT_STREAM_STOP)
        except asyncio.QueueFull:
            pass

    async def _process_audio_frame(self, frame: RealtimeAudioFrame) -> None:
        transcript: TranscriptChunk | None = None
        if self.asr is not None and callable(getattr(self.asr, "push_pcm", None)):
            try:
                transcript = await self.asr.push_pcm(frame)
            except Exception as exc:
                await self.handle_asr_failure(exc)
                return
            if transcript is not None:
                await self._publish({"event": "transcript", "payload": transcript.to_dict()})
                self.fusion.accept_transcript(transcript)
                await self._route_transcript_to_interpretation(transcript)
        if self.turn is None or not callable(getattr(self.turn, "push_pcm", None)):
            return
        try:
            turn_candidates = await self.turn.push_pcm(frame)
        except (asyncio.TimeoutError, TimeoutError):
            await self._handle_turn_timeout(transcript, frame)
            return
        for candidate in turn_candidates:
            for fusion_event in self.fusion.accept_turn(candidate):
                tentative = self.controller.handle_candidate(self.session_state, fusion_event)
                if tentative:
                    await self.apply_controller_actions(tentative)
                if fusion_event.event == "USER_TURN_END_CANDIDATE":
                    transcript = await self._finalize_turn_transcript(transcript)
                    await self._apply_policy(fusion_event, transcript)
                elif fusion_event.event == "USER_BACKCHANNEL_CANDIDATE":
                    await self._apply_policy(fusion_event, transcript)

    async def _finalize_turn_transcript(
        self, transcript: TranscriptChunk | None
    ) -> TranscriptChunk | None:
        if self.asr is None or not callable(getattr(self.asr, "finalize_turn", None)):
            return transcript
        try:
            final_chunk = await self.asr.finalize_turn()
        except Exception as exc:
            await self.handle_asr_failure(exc)
            return transcript
        if final_chunk.text or final_chunk.is_final:
            await self._publish({"event": "transcript", "payload": final_chunk.to_dict()})
            self.fusion.accept_transcript(final_chunk)
            await self._route_transcript_to_interpretation(final_chunk)
            return final_chunk
        return transcript

    async def _apply_policy(
        self,
        candidate: Any,
        transcript: TranscriptChunk | None,
    ) -> None:
        if self.conversation_mode is ConversationMode.INTERPRETATION:
            return
        if transcript is None and candidate.event != "USER_BACKCHANNEL_CANDIDATE":
            return
        if self.policy_engine is None:
            if transcript is not None and transcript.text.strip():
                await self._start_generation(transcript.text.strip())
            return
        request = PolicyRequest(
            state=self.session_state,
            assistant_last_text=self._assistant_last_text,
            unplayed_text_summary=self._unplayed_text_summary(),
            user_transcript=transcript,
            candidate=candidate,
        )
        decision = await self.policy_engine.decide(request)
        if "timeout" in decision.rationale.lower() and decision.action.value == "UNCERTAIN":
            await self.handle_policy_timeout(decision.rationale)
            return
        actions = self.controller.apply_policy(self.session_state, decision)
        if actions:
            await self.apply_controller_actions(actions)
        if any(action.action_type is ActionType.PROCESS_USER_REQUEST for action in actions):
            text = transcript.text.strip() if transcript is not None else ""
            if text:
                await self._start_generation(text)

    async def _start_generation(self, text: str) -> None:
        await self._cancel_generation()
        await self._generation_slot.replace(self._run_generation(text))

    async def _run_generation(self, text: str) -> None:
        await self._reset_provider(self.llm)
        await self._reset_provider(self.tts)
        epoch = self.advance_generation()
        response_id = self._next_response_id()
        token = CancellationToken()
        self._generation_token = token
        self.activate_response(response_id)
        self.session_state.floor = FloorState.ASSISTANT
        self.session_state.response = ResponseState.GENERATING
        await self._publish(
            {
                "event": "set_epoch",
                "response_id": response_id,
                "generation_epoch": epoch,
                "payload": {
                    "response_id": response_id,
                    "generation_epoch": epoch,
                },
            }
        )
        queue: asyncio.Queue[TextSegment | ResponseStreamEnd] = asyncio.Queue(
            maxsize=self.tts_queue_capacity
        )
        pipeline = RealtimeResponsePipeline(
            llm=self.llm,
            segment_queue=queue,
            generation_clock=self.generation_clock,
            segmenter=self.segmenter,
            cancellation_token=token,
            response_id_factory=lambda: response_id,
            on_token=self._publish_token,
        )
        consumer = asyncio.create_task(self._consume_segments(queue, epoch, token))
        try:
            result = await pipeline.run(self.prompt_builder(text))
            self._assistant_last_text = result.text
            await consumer
            if not result.stale and not result.cancelled:
                await self._publish(
                    {
                        "event": "response_completed",
                        "response_id": response_id,
                        "generation_epoch": epoch,
                        "payload": {
                            "response_id": response_id,
                            "generation_epoch": epoch,
                            "text": result.text,
                        },
                    }
                )
        except asyncio.CancelledError:
            token.cancel()
            raise
        finally:
            if not consumer.done():
                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            if self._generation_token is token:
                self._generation_token = None

    async def _publish_token(self, token: TokenChunk) -> None:
        self.record_generated_text(token.response_id or "", token.text)
        await self._publish(token)

    async def _consume_segments(
        self,
        queue: asyncio.Queue[TextSegment | ResponseStreamEnd],
        epoch: int,
        token: CancellationToken,
    ) -> None:
        while True:
            item = await queue.get()
            try:
                if token.is_cancelled() or not self.generation_clock.is_current(epoch):
                    if isinstance(item, ResponseStreamEnd):
                        return
                    continue
                if isinstance(item, ResponseStreamEnd):
                    return
                self.record_response_segment(item)
                keep_running = await self._stream_tts_segment(item, epoch, token)
                if not keep_running:
                    return
            finally:
                queue.task_done()

    async def _start_tts_stream(self, segment: TextSegment, request_id: str) -> Any:
        options = {
            "request_id": request_id,
            "response_id": segment.response_id,
            "generation_epoch": segment.generation_epoch,
            "segment_id": segment.segment_id,
        }
        playback_attempt_id = self.playback.active_playback_attempt_id
        if playback_attempt_id is not None:
            options["playback_attempt_id"] = playback_attempt_id
        stream = self._call_supported(self.tts.stream_audio, segment.text, **options)
        if inspect.isawaitable(stream):
            return await stream
        return stream

    def _handle_playback_ack(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("playback_ack payload must be a mapping")
        response_id = payload.get("response_id")
        generation_epoch = payload.get("generation_epoch")
        segment_id = payload.get("segment_id")
        sample_offset = payload.get("sample_offset")
        audio_time = payload.get("audio_time", 0.0)
        playback_attempt_id = payload.get("playback_attempt_id")
        if not all(isinstance(value, str) and value for value in (response_id,)):
            raise ValueError("playback_ack requires response_id")
        for name, value in {
            "generation_epoch": generation_epoch,
            "segment_id": segment_id,
            "sample_offset": sample_offset,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"playback_ack requires integer {name}")
        self.ack_playback(
            response_id,
            generation_epoch=generation_epoch,
            segment_id=segment_id,
            sample_offset=sample_offset,
            audio_time=float(audio_time),
            playback_attempt_id=playback_attempt_id if isinstance(playback_attempt_id, int) else None,
        )

    async def _cancel_generation_hook(self, *_args: Any) -> None:
        await self._cancel_generation()

    async def _cancel_generation(self) -> None:
        token = self._generation_token
        if token is not None:
            token.cancel()
        await self._generation_slot.cancel()
        await self._interrupt_provider(self.llm)
        await self._interrupt_provider(self.tts)
        self._playback_failed = False

    async def _publish_command(self, command: Any) -> None:
        normalized = _normalize_command_event(
            command,
            response_id=self.current_response_id,
            generation_epoch=self.generation_epoch,
        )
        if normalized is not None:
            await self._publish(normalized)

    async def _publish(self, value: Any) -> None:
        if self._closed_stream:
            return
        try:
            self._events.put_nowait(value)
        except asyncio.QueueFull as exc:
            self._terminal_error = SlowConsumerError("outbound event queue is full")
            await self.close()
            raise self._terminal_error from exc

    async def _publish_audio_chunk(self, chunk: AudioChunk) -> bool:
        if self._playback_failed:
            return False
        await self._publish(
            {
                "event": "audio_chunk",
                "response_id": chunk.response_id,
                "generation_epoch": chunk.generation_epoch,
                "segment_id": chunk.segment_id,
                "playback_attempt_id": chunk.playback_attempt_id,
                "payload": {
                    **chunk.to_dict(),
                    "sample_rate": self.audio_output_sample_rate,
                    "channels": self.audio_output_channels,
                },
            }
        )
        return not self._playback_failed

    async def _route_transcript_to_interpretation(self, transcript: TranscriptChunk) -> None:
        if self.conversation_mode is not ConversationMode.INTERPRETATION:
            return
        if not _is_committed_transcript_chunk(transcript):
            return
        await self.accept_transcript_chunk(transcript)

    def _unplayed_text_summary(self) -> str:
        if self.current_response_id is None:
            return ""
        checkpoint = self.checkpoints.get(self.current_response_id)
        if checkpoint is None:
            return ""
        return checkpoint.generated_text

    def snapshot(self) -> dict[str, Any]:
        current = self.checkpoints.get(self.current_response_id) if self.current_response_id else None
        paused = [
            self._checkpoint_snapshot(checkpoint)
            for checkpoint in self.checkpoints.all()
            if checkpoint.paused and checkpoint.resumable and not checkpoint.archived
        ]
        return {
            "mode": self.conversation_mode.value,
            "floor": self.session_state.floor.value,
            "response": self.session_state.response.value,
            "generation_epoch": self.generation_epoch,
            "current_response_id": self.current_response_id,
            "current_response": self._checkpoint_snapshot(current) if current is not None else None,
            "paused_responses": paused,
            "playback_cursor": self._playback_cursor(current),
        }

    def _next_response_id(self) -> str:
        self._response_identifiers += 1
        return f"response-{self._response_identifiers}"

    async def handle_policy_timeout(self, rationale: str = "policy timeout") -> None:
        self.session_state.floor = FloorState.USER
        self.session_state.response = ResponseState.PAUSED
        await self._publish({"event": "policy_uncertain", "payload": {"message": rationale}})
        await self._publish(
            {
                "event": "request_clarification",
                "payload": {"message": "I need you to repeat or clarify that request."},
            }
        )

    async def handle_asr_failure(self, error: BaseException) -> None:
        self.session_state.floor = FloorState.USER
        self.session_state.response = ResponseState.PAUSED
        await self._publish(
            {
                "event": "request_repeat",
                "payload": {"message": f"ASR failed, please repeat: {error}"},
            }
        )

    async def handle_playback_failure(self, error: BaseException) -> None:
        self._playback_failed = True
        self.session_state.response = ResponseState.PAUSED
        await self._publish(
            {
                "event": "playback_failed",
                "payload": {"message": str(error)},
            }
        )

    async def _handle_turn_timeout(
        self,
        transcript: TranscriptChunk | None,
        frame: RealtimeAudioFrame,
    ) -> None:
        activity = self.ingress.last_activity_candidate
        if activity is not None:
            self.fusion.accept_activity(activity)
        if transcript is None or not transcript.text.strip():
            return
        fallback = SpeechCandidateEvent(
            event="USER_TURN_END_CANDIDATE",
            event_id=f"turn-timeout-{frame.sequence}",
            timestamp=frame.capture_timestamp,
            source="turn_timeout_fallback",
            payload={
                "label": "turn_end",
                "confidence": 0.0,
                "evidence": {
                    "activity_active": bool(activity.active) if activity is not None else None,
                    "transcript": transcript.to_dict(),
                },
            },
        )
        await self._apply_policy(fallback, transcript)

    async def _stream_tts_segment(
        self,
        segment: TextSegment,
        epoch: int,
        token: CancellationToken,
    ) -> bool:
        request_id = self._tts_request_id(segment)
        published_any = False
        while True:
            try:
                stream = await self._start_tts_stream(segment, request_id)
                async for raw_chunk in _iterate_maybe_async(stream):
                    if token.is_cancelled() or not self.generation_clock.is_current(epoch):
                        return False
                    chunk = self._tag_audio(
                        raw_chunk,
                        segment,
                        request_id,
                        self.playback.active_playback_attempt_id,
                    )
                    self.record_audio_chunk(chunk)
                    self.session_state.response = ResponseState.PLAYING
                    self.session_state.floor = FloorState.ASSISTANT
                    published_any = True
                    if not await self._publish_audio_chunk(chunk):
                        return False
                return True
            except Exception as exc:
                if published_any:
                    terminal = await self._tts_supervisor.recover(exc)
                else:
                    terminal = await self._tts_supervisor.recover(
                        exc,
                        on_restart=self._restart_tts_worker,
                    )
                if terminal is True and not published_any:
                    continue
                await self._handle_terminal_tts_failure(segment, exc, terminal)
                return False

    async def _restart_tts_worker(self, _error: BaseException, _restart_count: int) -> None:
        await self._close_tts_runtime()
        await self._reset_provider(self.tts)

    async def _handle_terminal_tts_failure(
        self,
        segment: TextSegment,
        error: BaseException,
        terminal: bool | WorkerTerminalEvent,
    ) -> None:
        self.session_state.response = ResponseState.PAUSED
        await self._publish(
            {
                "event": "tts_failed",
                "response_id": segment.response_id,
                "generation_epoch": segment.generation_epoch,
                "payload": {"message": str(error), "text": segment.text},
            }
        )
        if isinstance(terminal, WorkerTerminalEvent):
            event = terminal.to_dict()
            event["response_id"] = segment.response_id
            event["generation_epoch"] = segment.generation_epoch
            await self._publish(event)

    async def _close_tts_runtime(self) -> None:
        for candidate in (
            self.tts,
            getattr(self.tts, "provider", None),
            getattr(getattr(self.tts, "provider", None), "runtime", None),
            getattr(self.tts, "runtime", None),
        ):
            close = getattr(candidate, "close", None) if candidate is not None else None
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
                return

    @staticmethod
    def _checkpoint_snapshot(checkpoint: Any) -> dict[str, Any]:
        return {
            "response_id": checkpoint.response_id,
            "generation_epoch": checkpoint.generation_epoch,
            "text": checkpoint.generated_text,
            "generated_cursor": checkpoint.generated_cursor,
            "committed_cursor": checkpoint.committed_cursor,
            "synthesized_cursor": checkpoint.synthesized_cursor,
            "played_cursor": checkpoint.played_cursor,
            "paused": checkpoint.paused,
        }

    @staticmethod
    def _playback_cursor(checkpoint: Any | None) -> dict[str, Any]:
        if checkpoint is None:
            return {
                "response_id": None,
                "segment_id": None,
                "sample_offset": 0,
                "played_cursor": 0,
            }
        pending_segment_id = None
        pending_offset = 0
        for segment_id in sorted(checkpoint.segments):
            segment = checkpoint.segments[segment_id]
            if segment.played_offset < segment.sample_count or (segment.sample_count == 0 and segment.text):
                pending_segment_id = segment_id
                pending_offset = segment.played_offset
                break
        return {
            "response_id": checkpoint.response_id,
            "segment_id": pending_segment_id,
            "sample_offset": pending_offset,
            "played_cursor": checkpoint.played_cursor,
        }

    @staticmethod
    def _tts_request_id(segment: TextSegment) -> str:
        if segment.response_id is None or segment.generation_epoch is None:
            raise RuntimeError("TTS segment is missing realtime identity")
        return f"{segment.response_id}:{segment.generation_epoch}:{segment.segment_id}"

    @staticmethod
    def _tag_audio(
        chunk: Any,
        segment: TextSegment,
        request_id: str,
        playback_attempt_id: int | None,
    ) -> AudioChunk:
        if isinstance(chunk, bytes):
            chunk = AudioChunk(
                chunk_id=request_id,
                audio_data=chunk,
                timestamp=time.time(),
                is_final=False,
            )
        if not isinstance(chunk, AudioChunk):
            raise TypeError("TTS provider must yield AudioChunk or bytes")
        payload = {
            "request_id": request_id,
            "response_id": segment.response_id,
            "generation_epoch": segment.generation_epoch,
            "segment_id": segment.segment_id,
        }
        if playback_attempt_id is not None:
            payload["playback_attempt_id"] = playback_attempt_id
        return replace(chunk, **payload)

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
        supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return method(*supported_args, **supported_kwargs)


async def _iterate_maybe_async(stream: Any) -> AsyncIterator[Any]:
    if hasattr(stream, "__aiter__"):
        async for item in stream:
            yield item
        return
    for item in stream:
        yield item


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("translated_text", "generated_text", "text", "token"):
            text = value.get(key)
            if isinstance(text, str):
                return text
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError("translation output must contain text")


def _is_committed_transcript_chunk(chunk: TranscriptChunk) -> bool:
    return bool(chunk.text or chunk.committed_text or chunk.replaces_committed or chunk.is_final)


def _normalize_command_event(
    command: Any,
    *,
    response_id: str | None,
    generation_epoch: int,
) -> dict[str, Any] | None:
    if isinstance(command, Mapping):
        event = str(command.get("event", command.get("type", "message"))).strip()
        if not event:
            return None
        payload = dict(command.get("payload", command))
        if response_id is not None:
            payload.setdefault("response_id", response_id)
        payload.setdefault("generation_epoch", generation_epoch)
        return {"event": event.lower(), **payload, "payload": payload}
    if command == "DUCK":
        return {"event": "duck", "payload": {"response_id": response_id, "generation_epoch": generation_epoch}}
    if command == "RESTORE":
        return {"event": "restore", "payload": {"response_id": response_id, "generation_epoch": generation_epoch}}
    if command == "PAUSE":
        return {"event": "pause_response", "payload": {"response_id": response_id, "generation_epoch": generation_epoch}}
    if command == "STOP":
        return {"event": "stop_response", "payload": {"response_id": response_id, "generation_epoch": generation_epoch}}
    return None


def _default_prompt_builder(text: str) -> str:
    return (
        "请回答用户的问题，回答要自然、简洁，适合直接转换成语音。\n"
        f"用户：{text}\n助手："
    )


class ProductionRealtimeRuntimeFactory:
    """Build per-session runtimes around shared heavyweight runtime resources."""

    def __init__(
        self,
        resource_builder: Callable[[], dict[str, Any]],
        session_builder: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        prompt_builder: Callable[[str], str] | None = None,
        session_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._resource_builder = resource_builder
        self._session_builder = session_builder
        self._resources: dict[str, Any] | None = None
        self.prompt_builder = prompt_builder or _default_prompt_builder
        self.session_options = dict(session_options or {})

    def __call__(self, session_id: str) -> ServerRealtimeSessionRuntime:
        resources = self._resources
        if resources is None:
            resources = self._resource_builder()
            self._resources = resources
        components = self._session_builder(resources)
        return ServerRealtimeSessionRuntime(
            session_id,
            llm=components["llm"],
            tts=components["tts"],
            asr=components.get("asr"),
            turn=components.get("turn"),
            policy_engine=components.get("policy_engine"),
            interpretation_translator=components.get("interpretation_translator"),
            prompt_builder=self.prompt_builder,
            **self.session_options,
        )

    async def close(self) -> None:
        resources = self._resources
        if resources is None:
            return
        for resource in resources.values():
            await _close_runtime_resource(resource)


def build_production_runtime_factory(
    *,
    model_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    runtime_loaders: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
    audio_config: Mapping[str, Any] | None = None,
    session_options: Mapping[str, Any] | None = None,
) -> ProductionRealtimeRuntimeFactory:
    """Compose the production realtime stack from configured model profiles."""
    effective = _effective_model_config(model_config, overrides or {})
    loaders = dict(_default_runtime_loaders())
    loaders.update(runtime_loaders or {})
    memory_manager = _build_model_manager(model_config)
    resolved_session_options = dict(session_options or {})
    if audio_config is not None:
        output_config = dict(audio_config.get("output", {}))
        resolved_session_options.setdefault(
            "audio_output_sample_rate",
            int(output_config.get("sample_rate", 24000)),
        )
        resolved_session_options.setdefault(
            "audio_output_channels",
            int(output_config.get("channels", 1)),
        )

    def resource_builder() -> dict[str, Any]:
        resources: dict[str, Any] = {}
        for name in ("asr", "turn", "policy", "llm", "tts", "translation"):
            profile = effective.get(name)
            if not profile:
                continue
            if memory_manager is not None:
                requested_bytes = _profile_int(
                    profile,
                    "estimated_vram_bytes",
                    default=0,
                )
                overhead_bytes = dict(profile.get("vram_overhead_bytes", {}))
                if requested_bytes or overhead_bytes:
                    memory_manager.reserve(
                        name,
                        requested_bytes=requested_bytes,
                        overhead_bytes={key: int(value) for key, value in overhead_bytes.items()},
                    )
            loaded = loaders[name](profile)
            if memory_manager is not None:
                used_bytes = _profile_int(profile, "loaded_vram_bytes", default=requested_bytes)
                if requested_bytes or used_bytes:
                    memory_manager.record_loaded(name, used_bytes=used_bytes)
            resources[name] = _SharedRuntimeBoundary(loaded)
        return resources

    def session_builder(resources: dict[str, Any]) -> dict[str, Any]:
        components: dict[str, Any] = {
            "llm": create_llm_adapter(
                effective["llm"],
                provider=_SessionRuntimeProxy(resources["llm"]),
            ),
            "tts": create_tts_adapter(
                effective["tts"],
                provider=_SessionRuntimeProxy(resources["tts"]),
            ),
        }
        if "asr" in resources:
            components["asr"] = create_realtime_asr_provider(
                effective["asr"],
                runtime=_SessionRuntimeProxy(resources["asr"]),
            )
        if "turn" in resources:
            components["turn"] = create_turn_provider(
                effective["turn"],
                runtime=_SessionRuntimeProxy(resources["turn"]),
            )
        if "policy" in resources:
            components["policy_engine"] = SemanticPolicyEngine(
                create_policy_provider(
                    effective["policy"],
                    provider=_SessionRuntimeProxy(resources["policy"]),
                )
            )
        translation_profile = effective.get("translation")
        if translation_profile and "translation" in resources:
            components["interpretation_translator"] = QwenTranslationAdapter(
                translation_profile["model_path"],
                runtime=_SessionRuntimeProxy(resources["translation"]),
                device=translation_profile.get("device", "cuda"),
                options=translation_profile.get("options"),
            )
        return components

    return ProductionRealtimeRuntimeFactory(
        resource_builder,
        session_builder,
        session_options=resolved_session_options,
    )


def _effective_model_config(
    model_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    models = model_config.get("models", model_config)
    result: dict[str, dict[str, Any]] = {}
    for name in ("asr", "turn", "policy", "llm", "tts", "translation"):
        profile = dict(models.get(name, {}))
        if not profile and name == "turn":
            profile = {
                "provider": "local",
                "model_id": "X2-Turn-4B-0812",
                "model_path": "./models/turn",
                "local_path": "./models/turn",
                "device": "cuda",
                "options": {
                    "cadence_ms": 160,
                    "context_seconds": 2.0,
                },
            }
        if not profile:
            continue
        model_override = overrides.get(f"{name}_model")
        if model_override:
            profile["model_path"] = str(model_override)
            profile["local_path"] = str(model_override)
        else:
            profile.setdefault("model_path", profile.get("local_path"))
        if name == "llm":
            profile["provider"] = "transformers"
        if name == "translation":
            profile["provider"] = "transformers"
        if name == "tts":
            profile["provider"] = "cosyvoice_worker"
            options = dict(profile.get("options", {}))
            for key in (
                "worker_python",
                "worker_script",
                "cosy_root",
                "prompt_audio",
                "prompt_text",
                "worker_startup_timeout",
            ):
                value = overrides.get(key)
                if value is not None:
                    option_name = "startup_timeout" if key == "worker_startup_timeout" else key
                    options[option_name] = value
            if options:
                profile["options"] = options
        result[name] = profile
    return result


def _load_x2_runtime_from_profile(profile: Mapping[str, Any]) -> Any:
    from src.adapters.turn.x2_turn_adapter import X2TurnAdapter

    options = dict(profile.get("options", {}))
    config = X2TurnConfig(
        model_path=str(profile.get("model_path", profile.get("local_path"))),
        device=str(profile.get("device", "cuda")),
        runtime_options=options,
    )
    return X2TurnAdapter(config=config).backend


def _default_runtime_loaders() -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    return {
        "asr": _load_faster_whisper_runtime_from_profile,
        "turn": _load_x2_runtime_from_profile,
        "policy": _load_qwen_policy_runtime_from_profile,
        "llm": _load_qwen_runtime_from_profile,
        "tts": _load_cosyvoice_runtime_from_profile,
        "translation": _load_qwen_runtime_from_profile,
    }


def _load_faster_whisper_runtime_from_profile(profile: Mapping[str, Any]) -> Any:
    from src.adapters.asr.providers.faster_whisper_streaming import _load_faster_whisper_runtime

    options = dict(profile.get("options", {}))
    return _load_faster_whisper_runtime(
        profile["model_path"],
        device=str(profile.get("device", "cuda")),
        compute_type=str(options.get("compute_type", "float16")),
    )


def _load_qwen_runtime_from_profile(profile: Mapping[str, Any]) -> Any:
    from src.adapters.llm.providers.qwen_transformers import TransformersQwenProvider

    provider = TransformersQwenProvider(
        profile["model_path"],
        device=str(profile.get("device", "cuda")),
        options=profile.get("options"),
    )
    return provider.runtime


def _load_qwen_policy_runtime_from_profile(profile: Mapping[str, Any]) -> Any:
    from src.adapters.llm.providers.qwen_policy import _load_transformers_runtime

    options = dict(profile.get("options", {}))
    return _load_transformers_runtime(
        profile.get("model_path"),
        str(profile.get("device", "cuda")),
        str(options.get("quantization", "none")),
    )


def _load_cosyvoice_runtime_from_profile(profile: Mapping[str, Any]) -> Any:
    from src.adapters.tts.providers.cosyvoice_worker import CosyVoiceWorkerClient

    return CosyVoiceWorkerClient(
        profile["model_path"],
        **dict(profile.get("options", {})),
    )


async def _close_runtime_resource(resource: Any) -> None:
    if resource is None:
        return
    for candidate in (
        resource,
        getattr(resource, "runtime", None),
        getattr(resource, "provider", None),
        getattr(getattr(resource, "provider", None), "runtime", None),
    ):
        close = getattr(candidate, "close", None) if candidate is not None else None
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return


def _build_model_manager(model_config: Mapping[str, Any]) -> ModelManager | None:
    runtime = dict(model_config.get("runtime", {}))
    budget = runtime.get("vram_budget_bytes")
    reserve = runtime.get("vram_reserve_bytes", 0)
    if budget is None:
        return None
    return ModelManager(
        total_vram_bytes=int(budget),
        reserve_bytes=int(reserve),
        measure_allocated_bytes=_measure_cuda_allocated_bytes,
    )


def _measure_cuda_allocated_bytes() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def _profile_int(profile: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    value = profile.get(key, default)
    return int(value)
