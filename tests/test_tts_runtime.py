import asyncio
import unittest

from src.adapters.tts.base import BaseTTSAdapter
from src.adapters.tts.backend import StreamingTTSBackend
from src.audio.frames import AudioFrame
from src.tts_runtime.pipeline import TTSRuntimePipeline
from src.tts_runtime.session import TTSSession, TTSSessionStatus
from src.tts_runtime.stream import AudioChunk
from src.llm_runtime.stream import TokenChunk


class FakeTTSAdapter(BaseTTSAdapter):
    def __init__(self, output=None):
        self.output = list(output or [b"audio"])
        self.texts = []
        self.interrupt_calls = 0

    async def synthesize(self, text):
        self.texts.append(text)

    async def stream_audio(self, text):
        return self.output

    async def interrupt(self):
        self.interrupt_calls += 1


class BlockingTTSAdapter(FakeTTSAdapter):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_audio(self, text):
        self.started.set()
        await self.release.wait()
        return [AudioChunk("late", b"stale", 1.0, False)]


class FakeProvider:
    def __init__(self):
        self.texts = []
        self.interrupted = False

    async def synthesize(self, text):
        self.texts.append(text)

    async def stream_audio(self, text):
        return [b"provider-audio", {"audio_data": b"", "is_final": True}]

    async def interrupt(self):
        self.interrupted = True


class TTSRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_lifecycle(self):
        created = TTSSession("session-1", TTSSessionStatus.CREATED, 1.0)
        self.assertEqual(created.status, TTSSessionStatus.CREATED)

        pipeline = TTSRuntimePipeline(FakeTTSAdapter())
        running = await pipeline.start_session("session-1")
        self.assertEqual(running.status, TTSSessionStatus.RUNNING)

        completed = await pipeline.complete_session()
        self.assertEqual(completed.status, TTSSessionStatus.COMPLETED)

    async def test_partial_audio_delivery_and_audio_frame_compatibility(self):
        chunks = [
            AudioChunk("one", b"one", 1.0, False),
            AudioChunk("two", b"two", 2.0, False),
        ]
        adapter = FakeTTSAdapter(chunks)
        pipeline = TTSRuntimePipeline(adapter)
        await pipeline.start_session("session-1")

        audio_chunks = await pipeline.push_text(TokenChunk("t1", "Hello", 1.0, False))
        frames = await pipeline.stream_audio()

        self.assertEqual(audio_chunks, tuple(chunks))
        self.assertEqual([frame.data for frame in frames], [b"one", b"two"])
        self.assertTrue(all(isinstance(frame, AudioFrame) for frame in frames))
        self.assertEqual(adapter.texts, ["Hello"])

    async def test_final_audio_chunk(self):
        adapter = FakeTTSAdapter([AudioChunk("final", b"", 2.0, True)])
        pipeline = TTSRuntimePipeline(adapter)
        await pipeline.start_session("session-1")

        chunks = await pipeline.push_text("done")

        self.assertTrue(chunks[-1].is_final)
        self.assertEqual(chunks[-1].audio_data, b"")

    async def test_interrupt_flushes_audio_queue_and_marks_session(self):
        adapter = FakeTTSAdapter([AudioChunk("queued", b"queued", 1.0, False)])
        pipeline = TTSRuntimePipeline(adapter)
        await pipeline.start_session("session-1")
        await pipeline.push_text("hello")

        interrupted = await pipeline.interrupt()

        self.assertEqual(interrupted.status, TTSSessionStatus.INTERRUPTED)
        self.assertEqual(adapter.interrupt_calls, 1)
        self.assertEqual(await pipeline.stream_audio(), ())

    async def test_no_stale_audio_after_interrupt(self):
        adapter = BlockingTTSAdapter()
        pipeline = TTSRuntimePipeline(adapter)
        await pipeline.start_session("session-1")

        push_task = asyncio.create_task(pipeline.push_text("hello"))
        await adapter.started.wait()
        await pipeline.interrupt()
        adapter.release.set()

        self.assertEqual(await push_task, ())
        self.assertEqual(await pipeline.stream_audio(), ())

    async def test_streaming_backend_wraps_fake_provider(self):
        provider = FakeProvider()
        pipeline = TTSRuntimePipeline(StreamingTTSBackend(provider))
        await pipeline.start_session("session-1")

        chunks = await pipeline.push_text("hello")
        await pipeline.interrupt()

        self.assertEqual([chunk.audio_data for chunk in chunks], [b"provider-audio", b""])
        self.assertTrue(provider.interrupted)


if __name__ == "__main__":
    unittest.main()
