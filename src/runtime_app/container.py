"""Dependency containers for legacy application and realtime runtime composition."""

from __future__ import annotations

import asyncio
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
from src.realtime.session_runtime import RealtimeSessionRuntime
from src.realtime.session_state import FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechEventFusion
from src.realtime.text_segmenter import LanguageAwareTextSegmenter, TextSegment
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler
from src.tts_runtime.stream import AudioChunk
from src.model_runtime.factory import (
    create_llm_adapter,
    create_policy_provider,
    create_realtime_asr_provider,
    create_tts_adapter,
    create_turn_provider,
)


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
        controller: ConversationController | None = None,
        fusion: SpeechEventFusion | None = None,
        prompt_builder: Callable[[str], str] | None = None,
        segmenter: LanguageAwareTextSegmenter | None = None,
        tts_queue_capacity: int = 2,
    ) -> None:
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.llm = llm
        self.tts = tts
        self.asr = asr
        self.turn = turn
        self.policy_engine = policy_engine
        self.controller = controller if controller is not None else ConversationController()
        self.prompt_builder = prompt_builder or _default_prompt_builder
        self.segmenter = segmenter or LanguageAwareTextSegmenter()
        self.tts_queue_capacity = tts_queue_capacity
        self.session_state = SessionState()
        self.fusion = fusion if fusion is not None else SpeechEventFusion(session_state=self.session_state)
        self._response_identifiers = 0
        self._connected = 0
        self._started = False
        self._closed_stream = False
        self._generation_slot = ActiveTaskSlot()
        self._generation_token: CancellationToken | None = None
        self._assistant_last_text = ""
        self._sender_playback = PlaybackCoordinator(send_command=self._publish_command)
        super().__init__(
            session_id,
            playback=self._sender_playback,
            cancel_generation=self._cancel_generation_hook,
        )

    async def start(self) -> None:
        self._started = True

    async def connect(self) -> None:
        self._connected += 1

    async def disconnect(self) -> None:
        self._connected = max(0, self._connected - 1)

    async def events(self) -> AsyncIterator[Any]:
        while True:
            item = await self._events.get()
            if item is _EVENT_STREAM_STOP:
                return
            yield item

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
        await self._events.put(_EVENT_STREAM_STOP)

    async def _process_audio_frame(self, frame: RealtimeAudioFrame) -> None:
        transcript: TranscriptChunk | None = None
        if self.asr is not None and callable(getattr(self.asr, "push_pcm", None)):
            transcript = await self.asr.push_pcm(frame)
            if transcript is not None:
                await self._publish({"event": "transcript", "payload": transcript.to_dict()})
                self.fusion.accept_transcript(transcript)
        if self.turn is None or not callable(getattr(self.turn, "push_pcm", None)):
            return
        for candidate in await self.turn.push_pcm(frame):
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
        final_chunk = await self.asr.finalize_turn()
        if final_chunk.text or final_chunk.is_final:
            await self._publish({"event": "transcript", "payload": final_chunk.to_dict()})
            self.fusion.accept_transcript(final_chunk)
            return final_chunk
        return transcript

    async def _apply_policy(
        self,
        candidate: Any,
        transcript: TranscriptChunk | None,
    ) -> None:
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
                request_id = self._tts_request_id(item)
                stream = await self._start_tts_stream(item, request_id)
                async for raw_chunk in _iterate_maybe_async(stream):
                    if token.is_cancelled() or not self.generation_clock.is_current(epoch):
                        break
                    chunk = self._tag_audio(raw_chunk, item, request_id, self.playback.active_playback_attempt_id)
                    self.record_audio_chunk(chunk)
                    self.session_state.response = ResponseState.PLAYING
                    self.session_state.floor = FloorState.ASSISTANT
                    await self._publish(chunk)
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
        await self._events.put(value)

    def _unplayed_text_summary(self) -> str:
        if self.current_response_id is None:
            return ""
        checkpoint = self.checkpoints.get(self.current_response_id)
        if checkpoint is None:
            return ""
        return checkpoint.generated_text

    def _next_response_id(self) -> str:
        self._response_identifiers += 1
        return f"response-{self._response_identifiers}"

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
    """Build per-session runtimes around one shared real-model adapter bundle."""

    def __init__(
        self,
        builder: Callable[[], dict[str, Any]],
        *,
        prompt_builder: Callable[[str], str] | None = None,
    ) -> None:
        self._builder = builder
        self._bundle: dict[str, Any] | None = None
        self.prompt_builder = prompt_builder or _default_prompt_builder

    def __call__(self, session_id: str) -> ServerRealtimeSessionRuntime:
        bundle = self._bundle
        if bundle is None:
            bundle = self._builder()
            self._bundle = bundle
        return ServerRealtimeSessionRuntime(
            session_id,
            llm=bundle["llm"],
            tts=bundle["tts"],
            asr=bundle.get("asr"),
            turn=bundle.get("turn"),
            policy_engine=bundle.get("policy_engine"),
            prompt_builder=self.prompt_builder,
        )

    async def close(self) -> None:
        bundle = self._bundle
        if bundle is None:
            return
        for resource in (
            bundle.get("turn"),
            bundle.get("asr"),
            bundle.get("policy_engine"),
            bundle.get("llm"),
            bundle.get("tts"),
        ):
            await _close_runtime_resource(resource)


def build_production_runtime_factory(
    *,
    model_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> ProductionRealtimeRuntimeFactory:
    """Compose the production realtime stack from configured model profiles."""
    effective = _effective_model_config(model_config, overrides or {})

    def builder() -> dict[str, Any]:
        llm = create_llm_adapter(effective["llm"])
        tts = create_tts_adapter(effective["tts"])
        asr = create_realtime_asr_provider(effective["asr"])
        turn = create_turn_provider(effective["turn"], runtime=_load_x2_runtime())
        policy_provider = create_policy_provider(effective["policy"])
        return {
            "llm": llm,
            "tts": tts,
            "asr": asr,
            "turn": turn,
            "policy_engine": SemanticPolicyEngine(policy_provider),
        }

    return ProductionRealtimeRuntimeFactory(builder)


def _effective_model_config(
    model_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    models = model_config.get("models", model_config)
    result: dict[str, dict[str, Any]] = {}
    for name in ("asr", "turn", "policy", "llm", "tts"):
        profile = dict(models.get(name, {}))
        if not profile and name == "turn":
            profile = {
                "provider": "local",
                "model_id": "X2-Turn-4B-0812",
                "model_path": "./models/turn",
                "local_path": "./models/turn",
                "device": "cuda",
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


def _load_x2_runtime() -> Any:
    from src.adapters.turn.x2_turn_adapter import X2TurnAdapter

    return X2TurnAdapter().backend


async def _close_runtime_resource(resource: Any) -> None:
    if resource is None:
        return
    for candidate in (
        resource,
        getattr(resource, "provider", None),
        getattr(getattr(resource, "provider", None), "runtime", None),
        getattr(resource, "provider", None),
    ):
        close = getattr(candidate, "close", None) if candidate is not None else None
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return
