import asyncio
import unittest

from src.core.interfaces.translation import TranslationAdapter
from src.realtime.cancellation import CancellationToken
from src.realtime.identifiers import GenerationClock
from src.realtime.interpretation import (
    InterpretationPipeline,
    TranslationSegment,
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
    def __init__(self):
        self.items = []

    async def publish(self, item: TranslationSegment):
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


def _extract_source_text(prompt: str) -> str:
    marker = "Source text:\n"
    _, _, suffix = prompt.partition(marker)
    return suffix


if __name__ == "__main__":
    unittest.main()
