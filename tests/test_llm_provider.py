import asyncio
import unittest

from benchmarks.metrics import calculate_metrics
from benchmarks.timeline import BenchmarkTimeline
from src.adapters.llm.providers.llama_cpp import LlamaCppLLMProvider
from src.llm_runtime.stream import TokenChunk
from src.model_runtime.factory import create_llm_adapter


class FakeLlamaRuntime:
    def __init__(self):
        self.cancelled = False
        self.prompts = []

    async def stream_tokens(self, prompt, **options):
        self.prompts.append(prompt)
        async def tokens():
            yield TokenChunk("one", "Hel", 1.0, False)
            if not self.cancelled:
                yield TokenChunk("final", "Hello world", 2.0, True)
        return tokens()

    async def cancel(self):
        self.cancelled = True


class BlockingLlamaRuntime(FakeLlamaRuntime):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def stream_tokens(self, prompt, **options):
        async def tokens():
            self.started.set()
            await self.release.wait()
            yield TokenChunk("late", "stale", 3.0, False)
        return tokens()


class LLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_creates_llama_provider(self):
        runtime = FakeLlamaRuntime()
        adapter = create_llm_adapter(
            {
                "provider": "llama_cpp",
                "model_path": "models/llm",
                "provider_runtime": runtime,
                "device": "cpu",
                "options": {"context_length": 4096},
            }
        )
        self.assertIsInstance(adapter.provider, LlamaCppLLMProvider)
        self.assertEqual(adapter.model_path, "models/llm")

    async def test_sync_and_async_token_stream_normalization(self):
        runtime = FakeLlamaRuntime()
        provider = LlamaCppLLMProvider("models/llm", runtime=runtime)
        result = await provider.stream_tokens("hello")
        chunks = [chunk async for chunk in result]
        self.assertEqual([chunk.text for chunk in chunks], ["Hel", "Hello world"])
        self.assertFalse(chunks[0].is_final)
        self.assertTrue(chunks[1].is_final)

    async def test_cancellation_discards_late_tokens(self):
        runtime = BlockingLlamaRuntime()
        provider = LlamaCppLLMProvider("models/llm", runtime=runtime)
        result = await provider.stream_tokens("hello")
        consume = asyncio.create_task(self._collect(result))
        await runtime.started.wait()
        await provider.cancel()
        runtime.release.set()
        self.assertEqual(await consume, [])

    async def test_missing_runtime_is_explicit(self):
        with self.assertRaises(RuntimeError):
            LlamaCppLLMProvider("models/llm")

    async def test_benchmark_ttft_and_cancellation_events(self):
        timeline = BenchmarkTimeline()
        timeline.record("turn_end", 1.0)
        timeline.record("first_llm_token", 1.12)
        timeline.record("cancel_requested", 2.0)
        timeline.record("generation_cancelled", 2.05)
        metrics = calculate_metrics(timeline)
        self.assertEqual(metrics["first_token_latency_ms"], 120.0)
        self.assertEqual(metrics["cancellation_latency_ms"], 50.0)

    @staticmethod
    async def _collect(result):
        return [chunk async for chunk in result]


if __name__ == "__main__":
    unittest.main()
