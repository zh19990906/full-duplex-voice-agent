import unittest

from src.adapters.asr.base import BaseASRAdapter
from src.asr.pipeline import ASRPipeline
from src.asr.session import ASRSession, ASRSessionStatus
from src.asr.stream import TranscriptChunk
from src.audio.frames import AudioFrame
from src.core.events.events import UserSpeechPartialEvent, UserTurnEndEvent


class FakeASRAdapter(BaseASRAdapter):
    def __init__(self, outputs=None):
        self.audio_chunks = []
        self.outputs = list(outputs or [])

    async def stream_audio(self, audio_chunk):
        self.audio_chunks.append(audio_chunk)
        if self.outputs:
            return self.outputs.pop(0)
        return None


class ASRPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_creation_and_lifecycle(self):
        created = ASRSession("session-1", "zh", ASRSessionStatus.CREATED)
        self.assertEqual(created.status, ASRSessionStatus.CREATED)

        pipeline = ASRPipeline(FakeASRAdapter())
        running = await pipeline.start_session("session-1", "zh")
        self.assertEqual(running.status, ASRSessionStatus.RUNNING)

        stopped = await pipeline.stop_session()
        self.assertEqual(stopped.status, ASRSessionStatus.STOPPED)
        completed = await pipeline.complete_session()
        self.assertEqual(completed.status, ASRSessionStatus.COMPLETED)

    async def test_partial_transcript_emits_partial_event(self):
        adapter = FakeASRAdapter([{"text": "你好", "is_final": False}])
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("session-1", "zh")

        events = await pipeline.push_audio(b"audio")

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], UserSpeechPartialEvent)
        self.assertEqual(events[0].payload["text"], "你好")

    async def test_final_transcript_emits_existing_turn_end_event(self):
        adapter = FakeASRAdapter([TranscriptChunk("chunk-1", "上海", 1.0, True)])
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("session-1", "zh")

        events = await pipeline.push_audio(
            AudioFrame("frame-1", 1.0, 16000, 1, b"audio")
        )

        self.assertIsInstance(events[0], UserTurnEndEvent)
        self.assertEqual(events[0].payload["text"], "上海")

    async def test_audio_streaming_input_reaches_fake_adapter(self):
        adapter = FakeASRAdapter()
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("session-1", "en")

        await pipeline.push_audio(AudioFrame("frame-1", 0.0, 16000, 1, b"one"))
        await pipeline.push_audio(b"two")

        self.assertEqual(adapter.audio_chunks, [b"one", b"two"])

    async def test_stop_cancels_audio_acceptance(self):
        adapter = FakeASRAdapter()
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("session-1", "zh")
        await pipeline.stop_session()

        with self.assertRaises(RuntimeError):
            await pipeline.push_audio(b"ignored")
        self.assertEqual(adapter.audio_chunks, [])


if __name__ == "__main__":
    unittest.main()
