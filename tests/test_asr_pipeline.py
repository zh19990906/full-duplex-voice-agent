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


def _apply_event_revisions(events):
    committed = ""
    display = ""
    for event in events:
        payload = event.payload
        if payload["replaces_committed"]:
            committed = payload["committed_text"]
        else:
            committed += payload["text"]
        display = committed + payload["unstable_text"]
    return display


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

    async def test_revision_aware_partial_fields_cross_pipeline(self):
        adapter = FakeASRAdapter([{
            "chunk_id": "rolling-7",
            "text": "北京",
            "timestamp": 3.5,
            "is_final": False,
            "revision_id": 7,
            "unstable_text": "旅游",
        }])
        pipeline = ASRPipeline(adapter)
        await pipeline.start_session("session-1", "zh")

        event = (await pipeline.push_audio(b"audio"))[0]

        self.assertEqual(
            event.payload,
            {
                "chunk_id": "rolling-7",
                "text": "北京",
                "is_final": False,
                "session_id": "session-1",
                "language": "zh",
                "revision_id": 7,
                "unstable_text": "旅游",
                "committed_text": None,
                "replaces_committed": False,
            },
        )

    async def test_final_authoritative_replacements_cross_pipeline_exactly(self):
        for old, replacement in (("old direction", "new request"), ("abcdef", "abc")):
            with self.subTest(replacement=replacement):
                adapter = FakeASRAdapter([[
                    {
                        "text": "",
                        "is_final": False,
                        "revision_id": 1,
                        "unstable_text": old,
                    },
                    {
                        "text": old,
                        "is_final": False,
                        "revision_id": 2,
                        "unstable_text": " tail",
                    },
                    {
                        "text": "",
                        "is_final": True,
                        "revision_id": 3,
                        "committed_text": replacement,
                        "replaces_committed": True,
                    },
                ]])
                pipeline = ASRPipeline(adapter)
                await pipeline.start_session("session-1", "en")

                events = await pipeline.push_audio(b"audio")

                self.assertIsInstance(events[-1], UserTurnEndEvent)
                self.assertEqual(events[-1].payload["revision_id"], 3)
                self.assertTrue(events[-1].payload["replaces_committed"])
                self.assertEqual(_apply_event_revisions(events), replacement)

    def test_mapping_normalization_validates_revision_fields(self):
        with self.assertRaises(ValueError):
            ASRPipeline._normalize_result({"text": "bad", "revision_id": -1})


if __name__ == "__main__":
    unittest.main()
