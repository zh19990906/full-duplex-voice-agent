import asyncio
import unittest

from src.asr.stream import TranscriptChunk
from src.core.interfaces.translation import TranslationAdapter
from src.realtime.cancellation import CancellationToken
from src.realtime.identifiers import GenerationClock
from src.realtime.interpretation import (
    InterpretationPipeline,
    TranslationSegment,
    build_correction_prompt,
    build_discard_correction_prompt,
    build_translation_prompt,
)


class RecordingTranslator(TranslationAdapter):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.source_texts = []

    async def translate_stream(self, text: str):
        self.prompts.append(text)
        self.source_texts.append(_extract_source_text(text))
        if not self.responses:
            raise AssertionError("unexpected translation request")
        return self.responses.pop(0)


class RecordingTranslationSink:
    def __init__(self, *, mark_started_ids=(), fail_first=0):
        self.items = []
        self.calls = []
        self.mark_started_ids = set(mark_started_ids)
        self.fail_first = fail_first

    async def publish(self, item: TranslationSegment):
        self.calls.append(item.translation_segment_id)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("sink failed")
        if item.translation_segment_id in self.mark_started_ids:
            item.playback_started = True
        self.items.append(item)


class BlockingTranslator(TranslationAdapter):
    def __init__(self):
        self.started = False
        self.release = None

    async def translate_stream(self, text: str):
        self.started = True
        if self.release is not None:
            await self.release.wait()
        return "late"


class SequencedBlockingTranslator(TranslationAdapter):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def translate_stream(self, text: str):
        self.prompts.append(text)
        if len(self.prompts) == 1:
            self.first_started.set()
            await self.release_first.wait()
        if not self.responses:
            raise AssertionError("unexpected translation request")
        return self.responses.pop(0)


class InterpretationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_interpretation_translates_each_stable_source_once(self):
        translator = RecordingTranslator(["I am going to Beijing tomorrow."])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_partial("我明天去北京")
        await pipeline.push_partial("我明天去北京")
        await pipeline.push_partial("我明天去北京")

        self.assertEqual(translator.source_texts, ["我明天去北京"])
        self.assertEqual([item.source_text for item in sink.items], ["我明天去北京"])
        self.assertEqual([item.source_segment_id for item in sink.items], [0])
        self.assertEqual([item.translation_segment_id for item in sink.items], [0])

    async def test_target_language_changes_apply_from_next_stable_source_segment_only(self):
        translator = RecordingTranslator(["Hello", "world"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_partial("你好")
        await pipeline.push_partial("你好")
        pipeline.set_target_language("French")
        await pipeline.push_partial("你好世界")
        await pipeline.push_partial("你好世界")

        self.assertEqual(
            [(item.source_text, item.target_language) for item in sink.items],
            [("你好", "English"), ("世界", "French")],
        )
        self.assertEqual([item.source_segment_id for item in sink.items], [0, 1])
        self.assertEqual([item.translation_segment_id for item in sink.items], [0, 1])

    async def test_started_translation_segment_is_not_rewritten_after_language_change(self):
        translator = RecordingTranslator(["Hello", "world"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_partial("你好")
        await pipeline.push_partial("你好")
        first = sink.items[0]
        pipeline.mark_translation_started(first.translation_segment_id)

        pipeline.set_target_language("French")
        await pipeline.push_partial("你好世界")
        await pipeline.push_partial("你好世界")

        self.assertEqual(first.translated_text, "Hello")
        self.assertEqual(first.target_language, "English")
        self.assertTrue(first.playback_started)
        self.assertEqual(translator.source_texts, ["你好", "世界"])

    def test_translation_prompt_is_deterministic_and_target_explicit(self):
        first = build_translation_prompt("我明天去北京", target_language="English")
        second = build_translation_prompt("我明天去北京", target_language="English")

        self.assertEqual(first, second)
        self.assertIn("Target language: English", first)
        self.assertIn("Source language: auto-detect", first)
        self.assertIn("Return only the translation text.", first)

    async def test_epoch_change_prevents_late_translation_publication(self):
        clock = GenerationClock()
        translator = BlockingTranslator()
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
            generation_clock=clock,
        )

        await pipeline.push_partial("你好")
        translator.release = asyncio.Event()
        task = asyncio.create_task(pipeline.push_partial("你好"))
        while not translator.started:
            await asyncio.sleep(0)
        clock.advance()
        translator.release.set()
        result = await task

        self.assertIsNone(result)
        self.assertEqual(sink.items, [])

    async def test_cancellation_token_prevents_late_translation_publication(self):
        translator = BlockingTranslator()
        sink = RecordingTranslationSink()
        cancellation = CancellationToken()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
            cancellation_token=cancellation,
        )

        await pipeline.push_partial("你好")
        translator.release = asyncio.Event()
        task = asyncio.create_task(pipeline.push_partial("你好"))
        while not translator.started:
            await asyncio.sleep(0)
        cancellation.cancel()
        translator.release.set()
        result = await task

        self.assertIsNone(result)
        self.assertEqual(sink.items, [])

    async def test_concurrent_committed_updates_publish_in_source_order(self):
        translator = SequencedBlockingTranslator(["first", "second"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        first = TranscriptChunk("chunk-1", "第一句", 1.0, False, revision_id=1)
        second = TranscriptChunk("chunk-2", "第二句", 2.0, False, revision_id=2)
        first_task = asyncio.create_task(pipeline.push_chunk(first))
        await translator.first_started.wait()
        second_task = asyncio.create_task(pipeline.push_chunk(second))
        await asyncio.sleep(0)
        translator.release_first.set()
        await asyncio.gather(first_task, second_task)

        self.assertEqual([item.source_text for item in sink.items], ["第一句", "第二句"])
        self.assertEqual([item.translation_segment_id for item in sink.items], [0, 1])

    async def test_authoritative_correction_replaces_not_started_translation(self):
        translator = RecordingTranslator(["first", "replacement"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_chunk(TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1))
        await pipeline.push_chunk(
            TranscriptChunk(
                "chunk-2",
                "",
                2.0,
                True,
                revision_id=2,
                committed_text="去北京",
                replaces_committed=True,
            )
        )

        self.assertEqual(
            [(item.translated_text, item.kind, item.supersedes_translation_segment_ids) for item in sink.items],
            [("first", "TRANSLATION", ()), ("replacement", "REPLACEMENT", (0,))],
        )

    async def test_authoritative_correction_appends_explicit_correction_after_started_translation(self):
        translator = RecordingTranslator(["first", "correction"])
        sink = RecordingTranslationSink(mark_started_ids={0})
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_chunk(TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1))
        await pipeline.push_chunk(
            TranscriptChunk(
                "chunk-2",
                "",
                2.0,
                True,
                revision_id=2,
                committed_text="去北京",
                replaces_committed=True,
            )
        )

        self.assertEqual(
            [(item.translated_text, item.kind, item.supersedes_translation_segment_ids) for item in sink.items],
            [("first", "TRANSLATION", ()), ("correction", "CORRECTION", (0,))],
        )
        self.assertTrue(sink.items[0].playback_started)

    async def test_authoritative_deletion_replaces_not_started_committed_state(self):
        translator = RecordingTranslator(["first", "second"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_chunk(TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1))
        deleted = await pipeline.push_chunk(
            TranscriptChunk(
                "chunk-2",
                "",
                2.0,
                True,
                revision_id=2,
                committed_text="",
                replaces_committed=True,
            )
        )
        appended = await pipeline.push_chunk(
            TranscriptChunk("chunk-3", "去北京", 3.0, False, revision_id=3)
        )

        self.assertIsNone(deleted)
        self.assertEqual(appended.source_segment_id, 1)
        self.assertEqual(appended.translation_segment_id, 1)
        self.assertEqual(pipeline.session.committed_source_text, "去北京")
        self.assertEqual(
            [item.source_text for item in pipeline.session.translation_segments],
            ["去上海", "去北京"],
        )

    async def test_authoritative_deletion_silently_removes_unplayed_translation(self):
        translator = RecordingTranslator(["first"])
        sink = RecordingTranslationSink()
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_chunk(TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1))
        deleted = await pipeline.push_chunk(
            TranscriptChunk(
                "chunk-2",
                "",
                2.0,
                True,
                revision_id=2,
                committed_text="",
                replaces_committed=True,
            )
        )

        self.assertIsNone(deleted)
        self.assertEqual([item.translated_text for item in sink.items], ["first"])
        self.assertEqual(pipeline.session.committed_source_text, "")

    async def test_authoritative_deletion_after_started_playback_appends_correction(self):
        translator = RecordingTranslator(["first", "disregard that"])
        sink = RecordingTranslationSink(mark_started_ids={0})
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )

        await pipeline.push_chunk(TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1))
        deleted = await pipeline.push_chunk(
            TranscriptChunk(
                "chunk-2",
                "",
                2.0,
                True,
                revision_id=2,
                committed_text="",
                replaces_committed=True,
            )
        )

        self.assertEqual(
            translator.prompts[-1],
            build_discard_correction_prompt(
                "去上海",
                target_language="English",
                source_language=None,
            ),
        )
        self.assertEqual(deleted.translated_text, "disregard that")
        self.assertEqual(deleted.kind, "CORRECTION")
        self.assertEqual(deleted.supersedes_translation_segment_ids, (0,))
        self.assertEqual(pipeline.session.committed_source_text, "")

    async def test_sink_failure_retries_same_segment_once_without_advancing_identity(self):
        translator = RecordingTranslator(["first"])
        sink = RecordingTranslationSink(fail_first=1)
        pipeline = InterpretationPipeline(
            translator=translator,
            sink=sink,
            target_language="English",
        )
        chunk = TranscriptChunk("chunk-1", "去上海", 1.0, False, revision_id=1)

        with self.assertRaises(RuntimeError):
            await pipeline.push_chunk(chunk)

        retried = await pipeline.push_chunk(chunk)

        self.assertEqual(translator.source_texts, ["去上海"])
        self.assertEqual(sink.calls, [0, 0])
        self.assertEqual([item.translation_segment_id for item in sink.items], [0])
        self.assertEqual(retried.translation_segment_id, 0)


def _extract_source_text(prompt: str) -> str:
    marker = "Source text:\n"
    _, _, suffix = prompt.partition(marker)
    return suffix


if __name__ == "__main__":
    unittest.main()
