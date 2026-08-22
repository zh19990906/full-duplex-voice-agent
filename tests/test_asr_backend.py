import asyncio
import unittest

from src.adapters.asr.backend import StreamingASRBackend
from src.asr.pipeline import ASRPipeline
from src.asr.stream import TranscriptChunk
from src.core.events.events import UserSpeechPartialEvent


class FakeProvider:
    def __init__(self):
        self.audio = []
        self.cancelled = False

    async def process_audio(self, audio_chunk):
        self.audio.append(audio_chunk)
        return [
            {"text": "hel", "is_final": False},
            {"text": "hello", "is_final": True},
        ]

    async def cancel(self):
        self.cancelled = True


class BlockingProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def process_audio(self, audio_chunk):
        self.audio.append(audio_chunk)
        self.started.set()
        await self.release.wait()
        return TranscriptChunk("late", "stale", 1.0, False)


class ASRBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_backend_initialization_and_fake_provider_compatibility(self):
        provider = FakeProvider()
        adapter = StreamingASRBackend(provider)

        result = await adapter.stream_audio(b"audio")

        self.assertEqual(provider.audio, [b"audio"])
        self.assertEqual(len(result), 2)

    async def test_partial_and_final_transcripts_stream(self):
        pipeline = ASRPipeline(StreamingASRBackend(FakeProvider()))
        await pipeline.start_session("session-1", "en")

        events = await pipeline.push_audio(b"audio")

        self.assertIsInstance(events[0], UserSpeechPartialEvent)
        self.assertEqual(events[0].payload["text"], "hel")
        self.assertTrue(events[1].payload["is_final"])

    async def test_backend_cancellation_stops_future_audio(self):
        provider = FakeProvider()
        adapter = StreamingASRBackend(provider)

        await adapter.stream_audio(b"before")
        await adapter.cancel()

        with self.assertRaises(RuntimeError):
            await adapter.stream_audio(b"after")
        self.assertTrue(provider.cancelled)

    async def test_pipeline_cancellation_drops_late_transcript(self):
        provider = BlockingProvider()
        pipeline = ASRPipeline(StreamingASRBackend(provider))
        await pipeline.start_session("session-1", "en")

        push_task = asyncio.create_task(pipeline.push_audio(b"audio"))
        await provider.started.wait()
        await pipeline.stop_session()
        provider.release.set()

        self.assertEqual(await push_task, ())
        self.assertEqual(pipeline.emitted_events, [])


if __name__ == "__main__":
    unittest.main()
