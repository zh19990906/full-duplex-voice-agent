"""Executable full-duplex integration validation using fake adapters."""

import asyncio
import time
from typing import Any

from src.adapters.asr.base import BaseASRAdapter
from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.tts.base import BaseTTSAdapter
from src.application.voice_agent import VoiceAgent
from src.audio.buffer import AudioBuffer
from src.audio.frames import AudioFrame
from src.audio.stream import AudioStream
from src.audio_pipeline.processor import AudioProcessor
from src.audio_pipeline.router import AudioRouter
from src.asr.pipeline import ASRPipeline
from src.controller.actions import ActionType
from src.controller.states import ControllerState
from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
)
from src.core.interfaces.translation import TranslationAdapter
from src.core.interfaces.turn import TurnAdapter
from src.generation.manager import GenerationManager
from src.integration.scenarios import ScenarioResult
from src.llm_runtime.pipeline import LLMGenerationPipeline
from src.llm_runtime.session import GenerationSessionStatus
from src.llm_runtime.stream import TokenChunk
from src.runtime_app.container import ApplicationContainer
from src.task.state_manager import TaskStateManager
from src.translation.pipeline import StreamingTranslationPipeline
from src.tts_runtime.pipeline import TTSRuntimePipeline
from src.tts_runtime.session import TTSSessionStatus


class DemoAudioStream(AudioStream):
    """In-memory audio input used by the integration demo."""

    def __init__(self) -> None:
        self._frames: asyncio.Queue[AudioFrame] = asyncio.Queue()

    async def read(self) -> AudioFrame:
        return await self._frames.get()

    async def write(self, frame: AudioFrame) -> None:
        await self._frames.put(frame)


class DemoASRAdapter(BaseASRAdapter):
    def __init__(self) -> None:
        self.received_audio: list[bytes] = []

    async def stream_audio(self, audio_chunk: bytes) -> dict[str, Any]:
        self.received_audio.append(audio_chunk)
        return {
            "text": audio_chunk.decode("utf-8"),
            "is_final": True,
            "timestamp": time.time(),
        }


class DemoTurnAdapter(TurnAdapter):
    def __init__(self, asr_pipeline: ASRPipeline) -> None:
        self.asr_pipeline = asr_pipeline

    async def push_audio(self, audio_chunk: bytes) -> None:
        await self.asr_pipeline.push_audio(audio_chunk)


class ControlledLLMAdapter(BaseLLMAdapter):
    """Fake streaming LLM that can be released or cancelled by a scenario."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def generate(self, prompt: str) -> str:
        return "Shanghai travel plan"

    async def stream_tokens(self, prompt: str):
        async def token_stream():
            yield TokenChunk("token-1", "I will explain", time.time(), False)
            self.started.set()
            await self.release.wait()
            if self.cancelled:
                return
            yield TokenChunk("token-2", " the plan", time.time(), False)
            yield TokenChunk("token-final", "", time.time(), True)

        return token_stream()

    async def cancel(self) -> None:
        self.cancelled = True
        self.release.set()

    def reset(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False


class DemoTTSAdapter(BaseTTSAdapter):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.interrupted = False
        self.interrupt_calls = 0

    async def synthesize(self, text: str) -> None:
        self.texts.append(text)

    async def stream_audio(self, text: str):
        if self.interrupted:
            return []
        return [text.encode("utf-8")]

    async def interrupt(self) -> None:
        self.interrupted = True
        self.interrupt_calls += 1

    def reset(self) -> None:
        self.interrupted = False


class DemoTranslationAdapter(TranslationAdapter):
    async def translate_stream(self, text: str) -> str:
        return "Hello"


class DemoTranslationTTSAdapter(BaseTTSAdapter):
    def __init__(self) -> None:
        self.streamed: list[str] = []

    async def synthesize(self, text: str) -> None:
        pass

    async def stream_audio(self, text: str) -> None:
        self.streamed.append(text)

    async def interrupt(self) -> None:
        pass


class FullDuplexIntegrationDemo:
    """Compose existing layers for deterministic end-to-end validation."""

    def __init__(self) -> None:
        self.application = ApplicationContainer()
        self.audio_stream = DemoAudioStream()
        self.audio_buffer = AudioBuffer()
        self.asr_adapter = DemoASRAdapter()
        self.last_asr_event: BaseEvent | None = None
        self.asr_pipeline = ASRPipeline(self.asr_adapter, self._handle_asr_event)
        self.turn_adapter = DemoTurnAdapter(self.asr_pipeline)
        self.audio_processor = AudioProcessor(
            self.audio_stream,
            self.audio_buffer,
            AudioRouter(self.turn_adapter),
        )
        self.llm_adapter = ControlledLLMAdapter()
        self.tts_adapter = DemoTTSAdapter()
        self.llm_pipeline = LLMGenerationPipeline(
            self.llm_adapter,
            self.application.generation_manager,
        )
        self.tts_pipeline = TTSRuntimePipeline(self.tts_adapter)
        self.task_manager = TaskStateManager()
        self._response_task: asyncio.Task[None] | None = None
        self.application.generation_manager.add_cancel_hook(self._on_generation_cancel)

    async def start(self) -> None:
        await self.application.initialize()
        await self.asr_pipeline.start_session("asr-demo", "en")
        await self.tts_pipeline.start_session("tts-demo")
        await self.audio_processor.start()

    async def shutdown(self) -> None:
        if self._response_task is not None and not self._response_task.done():
            await self._cancel_response()
        await self.audio_processor.stop()
        await self.application.shutdown()

    async def feed_audio(self, data: bytes) -> None:
        frame = AudioFrame("audio-demo", time.time(), 16000, 1, data)
        await self.audio_stream.write(frame)
        for _ in range(20):
            if (
                self.asr_adapter.received_audio
                and self.last_asr_event is not None
                and self.application.controller.state is ControllerState.THINKING
            ):
                return
            await asyncio.sleep(0)

    async def run_backchannel_scenario(self) -> ScenarioResult:
        self.application.controller.state = ControllerState.SPEAKING
        await self._start_response()
        await self.llm_adapter.started.wait()
        await self.application.event_bus.publish(self._backchannel_event())
        active = self.application.generation_manager.active_session is not None
        tts_untouched = not self.tts_adapter.interrupted
        self.llm_adapter.release.set()
        await self._wait_response()
        actions = [action.action_type for action in self.application.voice_agent.action_executor.history]
        passed = active and tts_untouched and ActionType.CONTINUE_GENERATION in actions
        return ScenarioResult(
            "backchannel_continuation",
            passed,
            {
                "generation_active_during_backchannel": active,
                "tts_interrupted": self.tts_adapter.interrupted,
                "continue_action": ActionType.CONTINUE_GENERATION in actions,
            },
        )

    async def run_interrupt_scenario(self) -> ScenarioResult:
        await self._start_response()
        await self.llm_adapter.started.wait()
        await self.application.event_bus.publish(self._interrupt_event())
        await self._wait_response()
        stale_frames = len(await self.tts_pipeline.stream_audio())
        actions = [action.action_type for action in self.application.voice_agent.action_executor.history]
        cancelled = self.llm_pipeline.session is not None and self.llm_pipeline.session.status is GenerationSessionStatus.CANCELLED
        interrupted = self.tts_pipeline.session is not None and self.tts_pipeline.session.status is TTSSessionStatus.INTERRUPTED
        passed = cancelled and interrupted and stale_frames == 0
        return ScenarioResult(
            "user_interruption",
            passed,
            {
                "generation_cancelled": cancelled,
                "tts_interrupted": interrupted,
                "stale_audio_frames": stale_frames,
                "stop_action": ActionType.STOP_RESPONSE in actions,
                "cancel_action": ActionType.CANCEL_GENERATION in actions,
            },
        )

    async def run_translation_scenario(self) -> ScenarioResult:
        translation_tts = DemoTranslationTTSAdapter()
        pipeline = StreamingTranslationPipeline(
            DemoTranslationAdapter(),
            translation_tts,
        )
        await pipeline.start_session("translation-demo", "zh", "en")
        chunk = await pipeline.push_text("你好", is_final=True)
        passed = chunk.translated_text == "Hello" and translation_tts.streamed == ["Hello"]
        return ScenarioResult(
            "streaming_translation",
            passed,
            {
                "translated_text": chunk.translated_text,
                "audio_frames": len(translation_tts.streamed),
            },
        )

    async def run_task_resume_scenario(self) -> ScenarioResult:
        task = self.task_manager.create_task(
            "count-demo",
            "counting",
            {"current_number": 5},
        )
        self.task_manager.pause_task(task.task_id)
        resumed = self.task_manager.resume_task(task.task_id)
        next_number = resumed.state_data["current_number"] + 1
        passed = resumed.state_data == {"current_number": 5}
        return ScenarioResult(
            "task_resume",
            passed,
            {
                "resumed_state": resumed.state_data,
                "next_number_contract": next_number,
            },
        )

    async def _handle_asr_event(self, event: BaseEvent) -> None:
        self.last_asr_event = event
        await self.application.event_bus.publish(event)

    async def _on_generation_cancel(self, _session: Any) -> None:
        if self.llm_pipeline.session is not None and self.llm_pipeline.session.status is GenerationSessionStatus.RUNNING:
            await self.llm_pipeline.cancel_generation()
        if self.tts_pipeline.session is not None and self.tts_pipeline.session.status is TTSSessionStatus.RUNNING:
            await self.tts_pipeline.interrupt()

    async def _start_response(self) -> None:
        if self._response_task is not None and not self._response_task.done():
            await self._cancel_response()
        await self.tts_pipeline.start_session(f"tts-{time.time_ns()}")
        await self.llm_pipeline.start_generation(f"llm-{time.time_ns()}", "travel plan")
        self._response_task = asyncio.create_task(self._stream_response())

    async def _stream_response(self) -> None:
        chunks = await self.llm_pipeline.stream_tokens()
        if self.llm_pipeline.session is None or self.llm_pipeline.session.status is not GenerationSessionStatus.RUNNING:
            return
        for chunk in chunks:
            if self.llm_pipeline.session.status is not GenerationSessionStatus.RUNNING:
                return
            await self.tts_pipeline.push_text(chunk)
        if self.llm_pipeline.session.status is GenerationSessionStatus.RUNNING:
            await self.llm_pipeline.complete_generation()
            await self.tts_pipeline.complete_session()

    async def _cancel_response(self) -> None:
        if self.application.generation_manager.active_session is not None:
            await self.application.generation_manager.cancel_current()
        await self._wait_response()

    async def _wait_response(self) -> None:
        if self._response_task is None:
            return
        await self._response_task
        self._response_task = None

    @staticmethod
    def _backchannel_event() -> UserBackchannelEvent:
        return UserBackchannelEvent("backchannel-1", time.time(), "demo", {"text": "嗯嗯，继续"})

    @staticmethod
    def _interrupt_event() -> UserInterruptEvent:
        return UserInterruptEvent("interrupt-1", time.time(), "demo", {"text": "Stop, Beijing"})
