"""Configuration-driven full-duplex composition demo.

This module wires existing layers only. Providers and device adapters are
injected by callers, which keeps the demo runnable with fake dependencies in
CI and ready for real deployment without changing runtime ownership.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.timeline import BenchmarkTimeline
from src.adapters.audio.microphone import MicrophoneAdapter
from src.adapters.audio.speaker import SpeakerAdapter
from src.adapters.asr.base import BaseASRAdapter
from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.tts.base import BaseTTSAdapter
from src.application.voice_agent import VoiceAgent
from src.audio.buffer import AudioBuffer
from src.audio.frames import AudioFrame
from src.audio_pipeline.processor import AudioProcessor
from src.audio_pipeline.router import AudioRouter
from src.asr.pipeline import ASRPipeline
from src.controller.states import ControllerState
from src.core.events.events import BaseEvent, UserInterruptEvent, UserTurnEndEvent
from src.core.interfaces.turn import TurnAdapter
from src.generation.manager import GenerationManager
from src.llm_runtime.pipeline import LLMGenerationPipeline
from src.llm_runtime.session import GenerationSessionStatus
from src.runtime_app.container import ApplicationContainer
from src.tts_runtime.pipeline import TTSRuntimePipeline
from src.tts_runtime.session import TTSSessionStatus


class _ASRTurnAdapter(TurnAdapter):
    def __init__(self, pipeline: ASRPipeline) -> None:
        self.pipeline = pipeline

    async def push_audio(self, audio_chunk: bytes) -> None:
        await self.pipeline.push_audio(audio_chunk)


def load_demo_config(path: str | Path = "configs/demo.yaml") -> dict[str, dict[str, str]]:
    """Load the intentionally small demo YAML contract without dependencies."""

    result: dict[str, dict[str, str]] = {"audio": {}, "models": {}}
    section: str | None = None
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            raise ValueError(f"invalid demo config at line {line_number}")
        key, value = (part.strip() for part in raw_line.strip().split(":", 1))
        if indentation == 0 and key in result and not value:
            section = key
        elif indentation >= 2 and section and value:
            result[section][key] = value.strip('"').strip("'")
        else:
            raise ValueError(f"invalid demo config indentation at line {line_number}")
    return result


class VoiceAgentDemo:
    """Compose microphone, speech runtimes, application, and speaker."""

    def __init__(
        self,
        microphone: MicrophoneAdapter,
        speaker: SpeakerAdapter,
        asr_adapter: BaseASRAdapter,
        llm_adapter: BaseLLMAdapter,
        tts_adapter: BaseTTSAdapter,
        application: ApplicationContainer | None = None,
        timeline: BenchmarkTimeline | None = None,
    ) -> None:
        self.microphone = microphone
        self.speaker = speaker
        self.application = application or ApplicationContainer()
        self.timeline = timeline
        self.asr_pipeline = ASRPipeline(asr_adapter, self._handle_asr_event)
        self.audio_processor = AudioProcessor(
            microphone,
            AudioBuffer(),
            AudioRouter(_ASRTurnAdapter(self.asr_pipeline)),
        )
        self.llm_pipeline = LLMGenerationPipeline(
            llm_adapter,
            self.application.generation_manager,
        )
        self.tts_pipeline = TTSRuntimePipeline(tts_adapter)
        self._response_task: asyncio.Task[None] | None = None
        self._started = False
        self.application.generation_manager.add_cancel_hook(self._cancel_hook)

    async def start(self) -> None:
        if self._started:
            return
        await self.application.initialize()
        await self.application.event_bus.subscribe(UserTurnEndEvent, self._handle_turn_end)
        await self.application.event_bus.subscribe(UserInterruptEvent, self._handle_interrupt_marker)
        await self.asr_pipeline.start_session("demo-asr", "auto")
        await self.tts_pipeline.start_session("demo-tts")
        await self.speaker.start()
        await self.audio_processor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self.application.generation_manager.active_session is not None:
            await self.application.generation_manager.cancel_current()
        if self._response_task is not None:
            await self._response_task
            self._response_task = None
        await self.audio_processor.stop()
        await self.asr_pipeline.stop_session()
        if self.tts_pipeline.session is not None and self.tts_pipeline.session.status is TTSSessionStatus.RUNNING:
            await self.tts_pipeline.interrupt()
        await self.speaker.stop()
        await self.application.event_bus.unsubscribe(UserTurnEndEvent, self._handle_turn_end)
        await self.application.event_bus.unsubscribe(UserInterruptEvent, self._handle_interrupt_marker)
        await self.application.shutdown()
        self._started = False

    async def _handle_asr_event(self, event: BaseEvent) -> None:
        if self.timeline is not None and event.event == "USER_SPEECH_PARTIAL":
            if self.timeline.first("first_asr_partial") is None:
                self.timeline.record("first_asr_partial", event.timestamp)
        await self.application.event_bus.publish(event)

    async def _handle_turn_end(self, event: UserTurnEndEvent) -> None:
        if self.timeline is not None:
            self.timeline.record("turn_end", event.timestamp)
        prompt = str(event.payload.get("text", ""))
        if self._response_task is not None and not self._response_task.done():
            await self.application.generation_manager.cancel_current()
        self._response_task = asyncio.create_task(self._generate_response(prompt))

    async def _handle_interrupt_marker(self, event: UserInterruptEvent) -> None:
        if self.timeline is not None:
            self.timeline.record("user_interrupt", event.timestamp)

    async def _generate_response(self, prompt: str) -> None:
        await self.tts_pipeline.start_session(f"tts-{time.time_ns()}")
        await self.llm_pipeline.start_generation(f"llm-{time.time_ns()}", prompt)
        chunks = await self.llm_pipeline.stream_tokens()
        for chunk in chunks:
            if self.llm_pipeline.session is None or self.llm_pipeline.session.status is not GenerationSessionStatus.RUNNING:
                return
            if self.timeline is not None and self.timeline.first("first_llm_token") is None:
                self.timeline.record("first_llm_token", chunk.timestamp)
            await self.tts_pipeline.push_text(chunk)
            for frame in await self.tts_pipeline.stream_audio():
                if self.timeline is not None and self.timeline.first("first_audio_chunk") is None:
                    self.timeline.record("first_audio_chunk", frame.timestamp)
                await self.speaker.write(frame)
        if self.llm_pipeline.session is not None and self.llm_pipeline.session.status is GenerationSessionStatus.RUNNING:
            await self.llm_pipeline.complete_generation()
            await self.tts_pipeline.complete_session()

    async def _cancel_hook(self, _session: Any) -> None:
        if self.timeline is not None:
            self.timeline.record("cancel_requested", time.time())
        if self.llm_pipeline.session is not None and self.llm_pipeline.session.status is GenerationSessionStatus.RUNNING:
            await self.llm_pipeline.cancel_generation()
        if self.tts_pipeline.session is not None and self.tts_pipeline.session.status is TTSSessionStatus.RUNNING:
            await self.tts_pipeline.interrupt()
        if self.timeline is not None:
            self.timeline.record("generation_cancelled", time.time())
            self.timeline.record("tts_stopped", time.time())


async def run_demo() -> list[Any]:
    """Run the deterministic existing integration scenarios as a smoke demo."""

    from src.integration.runner import run_all_scenarios

    return await run_all_scenarios()


def main() -> None:
    results = asyncio.run(run_demo())
    for result in results:
        print(f"{result.name}: {'PASS' if result.passed else 'FAIL'} {result.details}")
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
