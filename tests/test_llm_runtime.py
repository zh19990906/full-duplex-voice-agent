import asyncio
import unittest

from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.llm.backend import StreamingLLMBackend
from src.generation.manager import GenerationManager
from src.llm_runtime.pipeline import LLMGenerationPipeline
from src.llm_runtime.session import GenerationSession, GenerationSessionStatus
from src.llm_runtime.stream import TokenChunk


class FakeLLMAdapter(BaseLLMAdapter):
    def __init__(self, output=None):
        self.output = output if output is not None else []
        self.prompts = []
        self.cancel_calls = 0

    async def generate(self, prompt):
        return "".join(self.output)

    async def stream_tokens(self, prompt):
        self.prompts.append(prompt)
        return self.output

    async def cancel(self):
        self.cancel_calls += 1


class BlockingLLMAdapter(FakeLLMAdapter):
    def __init__(self):
        super().__init__()
        self.first_token = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_tokens(self, prompt):
        self.prompts.append(prompt)

        async def token_stream():
            yield "first"
            self.first_token.set()
            await self.release.wait()
            yield "second"

        return token_stream()


class FakeProvider:
    def __init__(self):
        self.prompts = []
        self.cancelled = False

    async def stream_tokens(self, prompt):
        self.prompts.append(prompt)
        return ["partial", {"text": "", "is_final": True}]

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return "complete"

    async def cancel(self):
        self.cancelled = True


class LLMRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_lifecycle(self):
        session = GenerationSession("session-1", GenerationSessionStatus.CREATED, 1.0)
        self.assertEqual(session.status, GenerationSessionStatus.CREATED)

        pipeline = LLMGenerationPipeline(FakeLLMAdapter())
        running = await pipeline.start_generation("session-1", "hello")
        self.assertEqual(running.status, GenerationSessionStatus.RUNNING)

        completed = await pipeline.complete_generation()
        self.assertEqual(completed.status, GenerationSessionStatus.COMPLETED)

    async def test_token_streaming_and_multiple_chunks(self):
        adapter = FakeLLMAdapter(["Hello", " world", "!"])
        pipeline = LLMGenerationPipeline(adapter)
        await pipeline.start_generation("session-1", "greet")

        chunks = await pipeline.stream_tokens()

        self.assertEqual([chunk.text for chunk in chunks], ["Hello", " world", "!"])
        self.assertEqual(adapter.prompts, ["greet"])

    async def test_final_token_handling(self):
        adapter = FakeLLMAdapter(
            [TokenChunk("chunk-1", "Hello", 1.0, False), TokenChunk("chunk-2", "", 2.0, True)]
        )
        pipeline = LLMGenerationPipeline(adapter)
        await pipeline.start_generation("session-1", "greet")

        chunks = await pipeline.stream_tokens()

        self.assertTrue(chunks[-1].is_final)
        self.assertEqual(chunks[-1].text, "")

    async def test_cancellation_marks_session_and_cleans_manager(self):
        manager = GenerationManager()
        adapter = FakeLLMAdapter(["Hello"])
        pipeline = LLMGenerationPipeline(adapter, manager)
        await pipeline.start_generation("session-1", "greet")

        cancelled = await pipeline.cancel_generation()

        self.assertEqual(cancelled.status, GenerationSessionStatus.CANCELLED)
        self.assertEqual(adapter.cancel_calls, 1)
        self.assertIsNone(manager.active_session)

        with self.assertRaises(RuntimeError):
            await pipeline.stream_tokens()

    async def test_cancellation_stops_active_token_stream(self):
        manager = GenerationManager()
        adapter = BlockingLLMAdapter()
        pipeline = LLMGenerationPipeline(adapter, manager)
        await pipeline.start_generation("session-1", "greet")

        stream_task = asyncio.create_task(pipeline.stream_tokens())
        await adapter.first_token.wait()
        await pipeline.cancel_generation()
        adapter.release.set()

        chunks = await stream_task

        self.assertEqual([chunk.text for chunk in chunks], ["first"])
        self.assertEqual(adapter.cancel_calls, 1)
        self.assertIsNone(manager.active_session)

    async def test_audio_independent_fake_adapter_integration(self):
        adapter = FakeLLMAdapter(["answer"])
        pipeline = LLMGenerationPipeline(adapter)
        await pipeline.start_generation("session-1", "question")

        await pipeline.stream_tokens()

        self.assertEqual(adapter.prompts, ["question"])

    async def test_streaming_backend_wraps_fake_provider(self):
        provider = FakeProvider()
        adapter = StreamingLLMBackend(provider)
        pipeline = LLMGenerationPipeline(adapter)
        await pipeline.start_generation("session-1", "question")

        chunks = await pipeline.stream_tokens()

        self.assertEqual([chunk.text for chunk in chunks], ["partial", ""])
        self.assertTrue(chunks[-1].is_final)
        self.assertEqual(provider.prompts, ["question"])

    async def test_backend_cancellation_stops_future_prompts(self):
        provider = FakeProvider()
        adapter = StreamingLLMBackend(provider)

        await adapter.stream_tokens("before")
        await adapter.cancel()

        with self.assertRaises(RuntimeError):
            await adapter.stream_tokens("after")
        self.assertTrue(provider.cancelled)

    async def test_new_session_drops_old_inflight_token_stream(self):
        manager = GenerationManager()
        adapter = BlockingLLMAdapter()
        pipeline = LLMGenerationPipeline(adapter, manager)
        await pipeline.start_generation("session-1", "first")

        old_stream = asyncio.create_task(pipeline.stream_tokens())
        await adapter.first_token.wait()
        await pipeline.start_generation("session-2", "second")
        adapter.release.set()

        old_chunks = await old_stream

        self.assertEqual([chunk.text for chunk in old_chunks], ["first"])


if __name__ == "__main__":
    unittest.main()
