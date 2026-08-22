import unittest

from src.core.interfaces.translation import TranslationAdapter
from src.core.interfaces.tts import TTSAdapter
from src.translation.pipeline import StreamingTranslationPipeline
from src.translation.session import TranslationSessionStatus
from src.translation.stream import TranslationChunk


class FakeTranslationAdapter(TranslationAdapter):
    def __init__(self):
        self.inputs = []

    async def translate_stream(self, text):
        self.inputs.append(text)
        return {"我们今天": "Today...", "出发": "depart"}.get(text, text)


class FakeTTSAdapter(TTSAdapter):
    def __init__(self):
        self.synthesized = []
        self.streamed = []

    async def synthesize(self, text):
        self.synthesized.append(text)

    async def stream_audio(self, text):
        self.streamed.append(text)

    async def interrupt(self):
        pass


class TranslationPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.translation = FakeTranslationAdapter()
        self.tts = FakeTTSAdapter()
        self.pipeline = StreamingTranslationPipeline(self.translation, self.tts)

    async def test_session_lifecycle(self):
        session = await self.pipeline.start_session("session-1", "zh", "en")
        self.assertEqual(session.status, TranslationSessionStatus.RUNNING)

        stopped = await self.pipeline.stop_session()
        self.assertEqual(stopped.status, TranslationSessionStatus.STOPPED)

        completed = await self.pipeline.complete_session()
        self.assertEqual(completed.status, TranslationSessionStatus.COMPLETED)

    async def test_translation_chunk_creation_and_serialization(self):
        chunk = TranslationChunk(
            chunk_id="chunk-1",
            source_text="我们今天",
            translated_text="Today...",
            timestamp=1.5,
            is_final=False,
        )

        self.assertEqual(chunk.to_dict(), {
            "chunk_id": "chunk-1",
            "source_text": "我们今天",
            "translated_text": "Today...",
            "timestamp": 1.5,
            "is_final": False,
        })

    async def test_streaming_input_calls_translation_and_tts(self):
        await self.pipeline.start_session("session-1", "zh", "en")

        partial = await self.pipeline.push_text("我们今天", is_final=False)
        final = await self.pipeline.push_text("出发", is_final=True)

        self.assertFalse(partial.is_final)
        self.assertTrue(final.is_final)
        self.assertEqual(self.translation.inputs, ["我们今天", "出发"])
        self.assertEqual(self.tts.streamed, ["Today...", "depart"])

    async def test_stop_and_complete_require_active_session(self):
        with self.assertRaises(RuntimeError):
            await self.pipeline.push_text("hello")
        with self.assertRaises(RuntimeError):
            await self.pipeline.stop_session()
        with self.assertRaises(RuntimeError):
            await self.pipeline.complete_session()


if __name__ == "__main__":
    unittest.main()
