import unittest

from src.realtime.stable_prefix import StablePrefixCommitter


class StablePrefixCommitterTests(unittest.TestCase):
    def test_committer_emits_only_new_common_prefix(self):
        committer = StablePrefixCommitter()

        self.assertEqual(committer.update("我想去北"), "")
        self.assertEqual(committer.update("我想去北京"), "我想去北")
        self.assertEqual(committer.update("我想去北京旅游"), "京")
        self.assertEqual(committer.committed_text, "我想去北京")
        self.assertEqual(committer.unstable_text, "旅游")

    def test_correction_shrink_and_unchanged_hypotheses_never_duplicate(self):
        committer = StablePrefixCommitter()
        self.assertEqual(committer.update("hello worl"), "")
        self.assertEqual(committer.update("hello work"), "hello wor")
        self.assertEqual(committer.update("hello wo"), "")
        self.assertEqual(committer.update("hello world"), "")
        self.assertEqual(committer.finalize("hello world"), "ld")
        self.assertEqual(committer.finalize("hello world"), "")
        self.assertEqual(committer.committed_text, "hello world")

    def test_unicode_empty_and_reset(self):
        committer = StablePrefixCommitter()
        self.assertEqual(committer.update("你好👋"), "")
        self.assertEqual(committer.update("你好👋世界"), "你好👋")
        self.assertEqual(committer.update(""), "")
        self.assertEqual(committer.unstable_text, "")
        committer.reset()
        self.assertEqual(committer.committed_text, "")
        self.assertEqual(committer.previous_hypothesis, "")
        self.assertEqual(committer.finalize("再见"), "再见")


if __name__ == "__main__":
    unittest.main()
