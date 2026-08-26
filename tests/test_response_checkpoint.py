import unittest

from src.realtime.checkpoint import ResponseCheckpointStore


class ResponseCheckpointStoreTests(unittest.TestCase):
    def test_cursors_advance_monotonically_and_pause_preserves_reusable_audio(self):
        store = ResponseCheckpointStore()
        store.activate("response-1", generation_epoch=0)

        store.record_generated_text("response-1", "北京是一座")
        store.record_generated_text("response-1", "历史悠久的城市。")
        store.record_segment(
            "response-1",
            0,
            "北京是一座历史悠久的城市。",
            audio=b"\x01\x00" * 8,
        )
        store.ack("response-1", segment_id=0, sample_offset=4)

        checkpoint = store.pause("response-1")
        plan = store.resume("response-1")

        self.assertEqual(checkpoint.generated_cursor, len("北京是一座历史悠久的城市。"))
        self.assertEqual(checkpoint.committed_cursor, len("北京是一座历史悠久的城市。"))
        self.assertEqual(checkpoint.synthesized_cursor, 8)
        self.assertEqual(checkpoint.played_cursor, 4)
        self.assertTrue(checkpoint.paused)
        self.assertEqual(plan.segment_id, 0)
        self.assertEqual(plan.sample_offset, 0)
        self.assertEqual(plan.audio, b"\x01\x00" * 8)

    def test_partial_segment_resumes_from_segment_start(self):
        store = ResponseCheckpointStore()
        store.activate("r1", generation_epoch=4)
        store.record_segment("r1", 4, "第二个是故宫。", audio=b"pcm")
        store.ack("r1", segment_id=4, sample_offset=1800)

        plan = store.resume("r1")

        self.assertEqual(plan.response_id, "r1")
        self.assertEqual(plan.generation_epoch, 4)
        self.assertEqual(plan.segment_id, 4)
        self.assertEqual(plan.sample_offset, 0)

    def test_pause_resume_preserves_audio_appended_in_multiple_chunks_for_one_segment(self):
        store = ResponseCheckpointStore()
        store.activate("response-2", generation_epoch=1)

        store.record_segment("response-2", 0, "先说这一句。", audio=b"\x01\x00" * 2)
        store.record_segment("response-2", 0, "先说这一句。", audio=b"\x02\x00" * 3)

        checkpoint = store.pause("response-2")
        plan = store.resume("response-2")

        self.assertEqual(checkpoint.synthesized_cursor, 5)
        self.assertEqual(plan.audio, (b"\x01\x00" * 2) + (b"\x02\x00" * 3))
        self.assertEqual(plan.segment_id, 0)
        self.assertEqual(plan.sample_offset, 0)


if __name__ == "__main__":
    unittest.main()
