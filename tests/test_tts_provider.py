import asyncio
import unittest

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline
from src.adapters.tts.providers.cosyvoice import CosyVoiceTTSProvider
from src.model_runtime.factory import create_tts_adapter
from src.tts_runtime.stream import AudioChunk


class FakeCosyVoiceRuntime:
    def __init__(self):
        self.interrupted = False
        self.texts = []

    async def stream_audio(self, text, **options):
        self.texts.append(text)
        async def chunks():
            yield AudioChunk("partial", b"part", 1.0, False)
            if not self.interrupted:
                yield AudioChunk("final", b"done", 2.0, True)
        return chunks()

    async def interrupt(self):
        self.interrupted = True


class BlockingCosyVoiceRuntime(FakeCosyVoiceRuntime):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def stream_audio(self, text, **options):
        async def chunks():
            self.started.set()
            await self.release.wait()
            yield AudioChunk("late", b"stale", 3.0, False)
        return chunks()


class TTSProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_creates_cosyvoice_provider(self):
        runtime = FakeCosyVoiceRuntime()
        adapter = create_tts_adapter(
            {
                "provider": "cosyvoice",
                "model_path": "models/tts",
                "provider_runtime": runtime,
                "device": "cpu",
                "options": {"speaker": "default"},
            }
        )
        self.assertIsInstance(adapter.provider, CosyVoiceTTSProvider)
        self.assertEqual(adapter.model_path, "models/tts")

    async def test_partial_and_final_audio_chunks(self):
        runtime = FakeCosyVoiceRuntime()
        provider = CosyVoiceTTSProvider("models/tts", runtime=runtime, speaker="default")
        result = await provider.stream_audio("hello")
        chunks = [chunk async for chunk in result]
        self.assertEqual([chunk.audio_data for chunk in chunks], [b"part", b"done"])
        self.assertFalse(chunks[0].is_final)
        self.assertTrue(chunks[1].is_final)

    async def test_interrupt_discards_late_audio(self):
        runtime = BlockingCosyVoiceRuntime()
        provider = CosyVoiceTTSProvider("models/tts", runtime=runtime)
        result = await provider.stream_audio("hello")
        consume = asyncio.create_task(self._collect(result))
        await runtime.started.wait()
        await provider.interrupt()
        runtime.release.set()
        self.assertEqual(await consume, [])

    async def test_missing_runtime_is_explicit(self):
        with self.assertRaises(RuntimeError):
            CosyVoiceTTSProvider("models/tts")

    def test_benchmark_ttfa_and_interrupt_latency(self):
        timeline = BenchmarkTimeline()
        timeline.record("first_llm_token", 1.0)
        timeline.record("first_audio_chunk", 1.2)
        timeline.record("user_interrupt", 2.0)
        timeline.record("tts_stopped", 2.08)
        metrics = calculate_metrics(timeline)
        self.assertEqual(metrics["first_audio_latency_ms"], 200.0)
        self.assertEqual(metrics["interrupt_latency_ms"], 80.0)

    @staticmethod
    async def _collect(result):
        return [chunk async for chunk in result]


if __name__ == "__main__":
    unittest.main()
