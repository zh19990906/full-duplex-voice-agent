import asyncio
import unittest

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline
from src.adapters.asr.base import BaseASRAdapter
from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.tts.base import BaseTTSAdapter
from src.asr.pipeline import ASRPipeline
from src.llm_runtime.pipeline import LLMGenerationPipeline
from src.tts_runtime.pipeline import TTSRuntimePipeline


class BlockingASRAdapter(BaseASRAdapter):
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def stream_audio(self, audio_chunk):
        self.started.set()
        await asyncio.Future()

    async def cancel(self):
        self.cancelled = True


class BlockingLLMAdapter(BaseLLMAdapter):
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def stream_tokens(self, prompt):
        self.started.set()
        await asyncio.Future()

    async def generate(self, prompt):
        return ""

    async def cancel(self):
        self.cancelled = True


class BlockingTTSAdapter(BaseTTSAdapter):
    def __init__(self):
        self.started = asyncio.Event()
        self.interrupted = False

    async def synthesize(self, text):
        return None

    async def stream_audio(self, text):
        self.started.set()
        await asyncio.Future()

    async def interrupt(self):
        self.interrupted = True


class PerformanceOptimizationTests(unittest.IsolatedAsyncioTestCase):
    def test_benchmark_includes_queue_and_startup_metrics(self):
        timeline = BenchmarkTimeline()
        for name, timestamp in (
            ("startup_started", 1.0),
            ("runtime_ready", 1.25),
            ("audio_frame_enqueued", 2.0),
            ("audio_frame_routed", 2.01),
            ("token_enqueued", 3.0),
            ("token_synthesized", 3.02),
        ):
            timeline.record(name, timestamp)

        metrics = calculate_metrics(timeline)

        self.assertEqual(metrics["startup_latency_ms"], 250.0)
        self.assertEqual(metrics["audio_buffer_wait_ms"], 10.0)
        self.assertEqual(metrics["token_queue_wait_ms"], 20.0)

    async def test_asr_stop_cancels_blocked_provider_call(self):
        adapter = BlockingASRAdapter()
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("asr-1", "en")
        push_task = asyncio.create_task(pipeline.push_audio(b"audio"))
        await adapter.started.wait()

        await pipeline.stop_session()

        self.assertEqual(await push_task, ())
        self.assertTrue(adapter.cancelled)

    async def test_llm_cancel_cancels_blocked_provider_call(self):
        adapter = BlockingLLMAdapter()
        pipeline = LLMGenerationPipeline(adapter)
        await pipeline.start_generation("llm-1", "hello")
        stream_task = asyncio.create_task(pipeline.stream_tokens())
        await adapter.started.wait()

        await pipeline.cancel_generation()

        self.assertEqual(await stream_task, ())
        self.assertTrue(adapter.cancelled)

    async def test_tts_interrupt_cancels_blocked_provider_call(self):
        adapter = BlockingTTSAdapter()
        pipeline = TTSRuntimePipeline(adapter)
        await pipeline.start_session("tts-1")
        push_task = asyncio.create_task(pipeline.push_text("hello"))
        await adapter.started.wait()

        await pipeline.interrupt()

        self.assertEqual(await push_task, ())
        self.assertTrue(adapter.interrupted)
        self.assertEqual(await pipeline.stream_audio(), ())


if __name__ == "__main__":
    unittest.main()
