import unittest

from src.realtime.text_segmenter import LanguageAwareTextSegmenter, TextSegment


class TextSegmentContractTests(unittest.TestCase):
    def test_text_segment_preserves_legacy_positional_arguments_and_transport_identity(self):
        segment = TextSegment(2, "第二句。", False, "response-7", 3)

        self.assertEqual(segment.segment_id, 2)
        self.assertEqual(
            segment.to_dict(),
            {
                "segment_id": 2,
                "text": "第二句。",
                "is_final": False,
                "response_id": "response-7",
                "generation_epoch": 3,
            },
        )

    def test_text_segment_rejects_empty_text_and_negative_identities(self):
        with self.assertRaises(ValueError):
            TextSegment(-1, "文本", False)
        with self.assertRaises(ValueError):
            TextSegment(0, "", False)
        with self.assertRaises(ValueError):
            TextSegment(0, "文本", False, "response", -1)


class LanguageAwareTextSegmenterTests(unittest.TestCase):
    def test_chinese_first_segment_commits_at_clause_boundary(self):
        segmenter = LanguageAwareTextSegmenter(first_min_chars=12)

        self.assertEqual(
            segmenter.push("北京是一座历史悠久的城市，"),
            (TextSegment(segment_id=0, text="北京是一座历史悠久的城市，", is_final=False),),
        )

    def test_chinese_comma_before_minimum_does_not_split(self):
        segmenter = LanguageAwareTextSegmenter(first_min_chars=12)

        self.assertEqual(segmenter.push("北京，"), ())
        self.assertEqual(
            segmenter.push("是一座历史悠久的城市。"),
            (TextSegment(0, "北京，是一座历史悠久的城市。", False),),
        )

    def test_english_waits_for_word_threshold_then_prefers_clause_boundary(self):
        segmenter = LanguageAwareTextSegmenter(
            first_min_words=3,
            next_min_words=5,
        )

        self.assertEqual(
            segmenter.push("Hello, wonderful world!"),
            (TextSegment(0, "Hello, wonderful world!", False),),
        )
        self.assertEqual(
            segmenter.push(" This is stable."),
            (),
        )
        self.assertEqual(
            segmenter.flush(),
            (TextSegment(1, " This is stable.", True),),
        )

    def test_hard_max_emits_without_punctuation(self):
        segmenter = LanguageAwareTextSegmenter(first_min_chars=100, first_max_chars=6)

        self.assertEqual(
            segmenter.push("甲乙丙丁戊己庚"),
            (TextSegment(0, "甲乙丙丁戊己", False),),
        )
        self.assertEqual(segmenter.flush(), (TextSegment(1, "庚", True),))

    def test_mixed_unicode_chunks_preserve_every_character_once(self):
        segmenter = LanguageAwareTextSegmenter(first_min_chars=4, next_min_chars=4)
        pieces = ("你好", "，world", "! 😀再", "见。")

        produced = [segment for piece in pieces for segment in segmenter.push(piece)]
        produced.extend(segmenter.flush())

        self.assertEqual("".join(segment.text for segment in produced), "你好，world! 😀再见。")
        self.assertTrue(produced[-1].is_final)
        self.assertEqual([segment.segment_id for segment in produced], list(range(len(produced))))

    def test_flush_is_idempotent_and_reset_restarts_segment_identity(self):
        segmenter = LanguageAwareTextSegmenter(first_min_chars=50)

        self.assertEqual(segmenter.push("未完成"), ())
        self.assertEqual(segmenter.flush(), (TextSegment(0, "未完成", True),))
        self.assertEqual(segmenter.flush(), ())
        segmenter.reset()
        self.assertEqual(segmenter.flush("新回答"), (TextSegment(0, "新回答", True),))


if __name__ == "__main__":
    unittest.main()
