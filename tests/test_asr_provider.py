import asyncio
import unittest

from src.adapters.asr.providers.whisper import WhisperASRProvider
from src.asr.stream import TranscriptChunk
from src.model_runtime.factory import create_asr_adapter


class FakeWhisperRuntime:
    def __init__(self):
        self.cancelled = False
        self.audio = []

    async def stream_audio(self, audio, **options):
        self.audio.append(audio)
        async def chunks():
            yield TranscriptChunk("partial", "hello", 1.0, False)
            if not self.cancelled:
                yield TranscriptChunk("final", "hello world", 2.0, True)
        return chunks()

    async def cancel(self):
        self.cancelled = True


class BlockingWhisperRuntime(FakeWhisperRuntime):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def stream_audio(self, audio, **options):
        async def stream():
            self.started.set()
            await self.release.wait()
            yield TranscriptChunk("late", "stale", 3.0, False)

        return stream()


class ASRProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_creates_whisper_provider(self):
        runtime = FakeWhisperRuntime()
        adapter = create_asr_adapter(
            {
                "provider": "whisper",
                "model_path": "models/asr",
                "provider_runtime": runtime,
                "device": "cpu",
            }
        )
        self.assertIsInstance(adapter.provider, WhisperASRProvider)
        self.assertEqual(adapter.model_path, "models/asr")

    async def test_partial_and_final_transcripts(self):
        runtime = FakeWhisperRuntime()
        provider = WhisperASRProvider("models/asr", runtime=runtime, language="auto")
        result = await provider.stream_audio(b"audio")
        chunks = [chunk async for chunk in result]
        self.assertEqual([chunk.text for chunk in chunks], ["hello", "hello world"])
        self.assertFalse(chunks[0].is_final)
        self.assertTrue(chunks[1].is_final)

    async def test_cancellation_discards_late_transcripts(self):
        runtime = FakeWhisperRuntime()
        provider = WhisperASRProvider("models/asr", runtime=runtime)
        await provider.cancel()
        with self.assertRaises(RuntimeError):
            await provider.stream_audio(b"late")
        self.assertEqual(runtime.audio, [])

    async def test_cancellation_discards_active_stream_callback(self):
        runtime = BlockingWhisperRuntime()
        provider = WhisperASRProvider("models/asr", runtime=runtime)
        result = await provider.stream_audio(b"audio")
        consume = asyncio.create_task(self._collect(result))
        await runtime.started.wait()
        await provider.cancel()
        runtime.release.set()
        self.assertEqual(await consume, [])

    @staticmethod
    async def _collect(result):
        return [chunk async for chunk in result]

    def test_missing_runtime_is_explicit(self):
        with self.assertRaises(RuntimeError):
            WhisperASRProvider("models/asr")


if __name__ == "__main__":
    unittest.main()
