import asyncio
import unittest

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline
from src.adapters.tts.providers.cosyvoice import CosyVoiceTTSProvider
from src.adapters.tts.providers.cosyvoice_worker import CosyVoiceWorkerClient
from src.llm_runtime.stream import TokenChunk
from src.model_runtime.factory import create_tts_adapter
from src.realtime.text_segmenter import LanguageAwareTextSegmenter
from src.runtime_app.real_session import RealModelSession
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


class FactoryWorkerRuntime:
    """External worker boundary used to test the real factory wrapper stack."""

    def __init__(self):
        self.calls = []
        self.cancelled = []
        self.interrupted = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_audio(self, text, **options):
        self.calls.append((text, options))

        async def chunks():
            self.started.set()
            await self.release.wait()
            yield AudioChunk("late", b"late", 1.0, True)

        return chunks()

    async def cancel(self, request_id):
        self.cancelled.append(request_id)

    async def interrupt(self):
        self.interrupted += 1


class TwoTokenLLM:
    async def stream_tokens(self, _prompt):
        yield TokenChunk("one", "Hello", 1.0, False)
        yield TokenChunk("two", " there", 2.0, True)

    async def cancel(self):
        return None


class TTSProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_worker_stack_forwards_identity_and_cancels_request(self):
        """Breaks if factory wrappers drop per-call identity or cancel(request_id)."""

        runtime = FactoryWorkerRuntime()
        tts = create_tts_adapter(
            {
                "provider": "cosyvoice_worker",
                "model_path": "models/tts",
                "provider_instance": runtime,
                "options": {"prompt_text": "configured prompt"},
            }
        )
        session = RealModelSession(
            TwoTokenLLM(),
            tts,
            lambda _value: None,
            segmenter=LanguageAwareTextSegmenter(first_min_words=1, next_min_words=1),
        )

        running = asyncio.create_task(session.run("question"))
        await asyncio.wait_for(runtime.started.wait(), timeout=0.5)
        await session.interrupt()
        runtime.release.set()
        await asyncio.wait_for(running, timeout=0.5)

        text, options = runtime.calls[0]
        self.assertEqual(text, "Hello")
        self.assertEqual(options["prompt_text"], "configured prompt")
        self.assertEqual(options["segment_id"], 0)
        self.assertEqual(options["response_id"], "response-1")
        self.assertEqual(options["generation_epoch"], 1)
        self.assertEqual(options["request_id"], "response-1:1:0")
        self.assertEqual(runtime.cancelled, ["response-1:1:0"])

    async def test_factory_worker_stack_resets_wrapped_provider_for_the_next_request(self):
        """Breaks if backend reset leaves the wrapped provider interrupted."""

        runtime = FactoryWorkerRuntime()
        tts = create_tts_adapter(
            {
                "provider": "cosyvoice_worker",
                "model_path": "models/tts",
                "provider_instance": runtime,
            }
        )
        session = RealModelSession(
            TwoTokenLLM(),
            tts,
            lambda _value: None,
            segmenter=LanguageAwareTextSegmenter(first_min_words=1, next_min_words=1),
        )

        first = asyncio.create_task(session.run("first"))
        await asyncio.wait_for(runtime.started.wait(), timeout=0.5)
        await session.interrupt()
        runtime.release.set()
        await asyncio.wait_for(first, timeout=0.5)

        self.assertEqual(await asyncio.wait_for(session.run("second"), timeout=0.5), "Hello there")

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

    async def test_factory_builds_isolated_cosyvoice_worker_from_config(self):
        adapter = create_tts_adapter(
            {
                "provider": "cosyvoice_worker",
                "model_path": "/mnt/models/cosyvoice",
                "options": {
                    "worker_python": "/home/CosyVoice/.venv/bin/python",
                    "prompt_audio": "/tmp/prompt.wav",
                    "prompt_text": "reference",
                },
            }
        )

        self.assertIsInstance(adapter.provider, CosyVoiceTTSProvider)
        self.assertIsInstance(adapter.provider.runtime, CosyVoiceWorkerClient)
        self.assertEqual(adapter.provider.runtime.prompt_audio, "/tmp/prompt.wav")

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
        timeline.record("turn_end", 0.8)
        timeline.record("first_llm_token", 1.0)
        timeline.record("first_audio_chunk", 1.2)
        timeline.record("user_interrupt", 2.0)
        timeline.record("tts_stopped", 2.08)
        metrics = calculate_metrics(timeline)
        self.assertEqual(metrics["first_audio_latency_ms"], 400.0)
        self.assertEqual(metrics["interrupt_latency_ms"], 80.0)

    @staticmethod
    async def _collect(result):
        return [chunk async for chunk in result]


if __name__ == "__main__":
    unittest.main()
