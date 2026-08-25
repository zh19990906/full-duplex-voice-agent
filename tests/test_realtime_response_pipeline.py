import asyncio
import unittest

from src.llm_runtime.stream import TokenChunk
from src.realtime.cancellation import CancellationToken
from src.realtime.identifiers import GenerationClock
from src.realtime.response_pipeline import (
    RealtimeResponsePipeline,
    ResponsePipelineResult,
    ResponseStreamEnd,
)
from src.realtime.text_segmenter import LanguageAwareTextSegmenter, TextSegment


class ControlledTokenProvider:
    def __init__(self, tokens, *, first_token_seen=None, release_after_first=None, failure=None):
        self.tokens = tuple(tokens)
        self.first_token_seen = first_token_seen
        self.release_after_first = release_after_first
        self.failure = failure
        self.cancelled = 0
        self.closed = 0

    async def stream_tokens(self, prompt):
        try:
            for index, token in enumerate(self.tokens):
                if index == 0 and self.first_token_seen is not None:
                    self.first_token_seen.set()
                if index == 1 and self.release_after_first is not None:
                    await self.release_after_first.wait()
                if self.failure is not None and index == self.failure:
                    raise RuntimeError("provider failed")
                yield token
        finally:
            self.closed += 1

    async def cancel(self):
        self.cancelled += 1
        if self.release_after_first is not None:
            self.release_after_first.set()


class RecordingSegmentQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class ResetAfterCancelProvider:
    def __init__(self):
        self.first_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.reset_calls = 0
        self.run_count = 0

    async def stream_tokens(self, prompt):
        if self.cancelled:
            raise RuntimeError("provider remains cancelled")
        self.run_count += 1
        if self.run_count == 1:
            self.first_seen.set()
            yield "旧"
            await self.release.wait()
            yield "回答。"
            return
        yield TokenChunk("fresh", "新回答。", 3.0, True)

    async def cancel(self):
        self.cancelled = True
        self.release.set()

    def reset(self):
        self.reset_calls += 1
        self.cancelled = False


class ResetProbeProvider:
    def __init__(self):
        self.reset_calls = 0
        self.cancel_calls = 0

    async def stream_tokens(self, prompt):
        yield TokenChunk("unused", "不应生成", 1.0, True)

    async def cancel(self):
        self.cancel_calls += 1

    def reset(self):
        self.reset_calls += 1


class RealtimeResponsePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_epoch_tokens_never_reach_tts_queue(self):
        clock = GenerationClock()
        first_token_seen = asyncio.Event()
        llm = ControlledTokenProvider(
            tokens=("旧", "回答", "。"),
            first_token_seen=first_token_seen,
            release_after_first=asyncio.Event(),
        )
        tts_queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=tts_queue,
            generation_clock=clock,
        )
        task = asyncio.create_task(pipeline.run("问题"))
        await first_token_seen.wait()
        clock.advance()
        llm.release_after_first.set()

        result = await task

        self.assertTrue(result.stale)
        self.assertEqual(tts_queue.items, [])
        self.assertEqual(llm.cancelled, 1)

    async def test_pipeline_tags_token_and_segment_outputs_with_one_run_identity(self):
        clock = GenerationClock()
        llm = ControlledTokenProvider((TokenChunk("a", "你好，", 1.0, False), "世界。"))
        queue = RecordingSegmentQueue()
        published = []
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=queue,
            generation_clock=clock,
            segmenter=LanguageAwareTextSegmenter(first_min_chars=2),
            on_token=published.append,
        )

        result = await pipeline.run("问题")

        self.assertIsInstance(result, ResponsePipelineResult)
        self.assertEqual(result.text, "你好，世界。")
        self.assertFalse(result.cancelled)
        self.assertFalse(result.stale)
        self.assertEqual([token.text for token in published], ["你好，", "世界。"])
        self.assertTrue(all(token.response_id == result.response_id for token in published))
        self.assertTrue(all(token.generation_epoch == result.generation_epoch for token in published))
        segments = [item for item in queue.items if isinstance(item, TextSegment)]
        self.assertEqual([item.text for item in segments], ["你好，", "世界。"])
        self.assertEqual([item.segment_id for item in segments], [0, 1])
        self.assertTrue(all(item.response_id == result.response_id for item in segments))

    async def test_final_token_marks_the_last_remaining_segment_final(self):
        queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=ControlledTokenProvider((TokenChunk("final", "最后一句。", 1.0, True),)),
            segment_queue=queue,
            generation_clock=GenerationClock(),
            segmenter=LanguageAwareTextSegmenter(first_min_chars=2),
        )

        await pipeline.run("问题")

        self.assertEqual(
            [
                (segment.text, segment.is_final)
                for segment in queue.items
                if isinstance(segment, TextSegment)
            ],
            [("最后一句。", True)],
        )
        self.assertIsInstance(queue.items[-1], ResponseStreamEnd)

    async def test_qwen_empty_final_sentinel_confirms_prior_punctuation_without_duplicate_tts_text(self):
        queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=ControlledTokenProvider(
                (
                    TokenChunk("punctuation", "你好。", 1.0, False),
                    TokenChunk("final", "", 2.0, True),
                )
            ),
            segment_queue=queue,
            generation_clock=GenerationClock(),
            segmenter=LanguageAwareTextSegmenter(first_min_chars=1),
        )

        result = await pipeline.run("问题")

        text_segments = [item for item in queue.items if isinstance(item, TextSegment)]
        endings = [item for item in queue.items if isinstance(item, ResponseStreamEnd)]
        self.assertEqual([(item.text, item.is_final) for item in text_segments], [("你好。", False)])
        self.assertEqual(len(endings), 1)
        self.assertEqual(endings[0].response_id, result.response_id)
        self.assertEqual(endings[0].generation_epoch, result.generation_epoch)
        self.assertEqual(endings[0].final_segment_id, 0)
        self.assertEqual(endings[0].status, "completed")

    async def test_empty_successful_response_emits_only_an_identity_bearing_terminal_marker(self):
        queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=ControlledTokenProvider((TokenChunk("final", "", 1.0, True),)),
            segment_queue=queue,
            generation_clock=GenerationClock(),
        )

        result = await pipeline.run("问题")

        self.assertEqual(len(queue.items), 1)
        self.assertIsInstance(queue.items[0], ResponseStreamEnd)
        self.assertEqual(queue.items[0].response_id, result.response_id)
        self.assertIsNone(queue.items[0].final_segment_id)

    async def test_epoch_change_while_queue_is_full_never_enqueues_stale_segment(self):
        clock = GenerationClock()
        queue = asyncio.Queue(maxsize=1)
        await queue.put(TextSegment(99, "旧占位", False))
        token_seen = asyncio.Event()
        llm = ControlledTokenProvider(("新的稳定句。",), first_token_seen=token_seen)
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=queue,
            generation_clock=clock,
            segmenter=LanguageAwareTextSegmenter(first_min_chars=2),
            capacity_poll_seconds=0.005,
        )
        task = asyncio.create_task(pipeline.run("问题"))
        await token_seen.wait()
        await asyncio.sleep(0.02)
        clock.advance()
        await queue.get()
        queue.task_done()

        result = await task

        self.assertTrue(result.stale)
        self.assertTrue(queue.empty())
        self.assertEqual(llm.cancelled, 1)

    async def test_late_tokens_and_callbacks_stop_after_cancellation(self):
        clock = GenerationClock()
        cancellation = CancellationToken()
        first_token_seen = asyncio.Event()
        llm = ControlledTokenProvider(
            ("第一句。", "第二句。"),
            first_token_seen=first_token_seen,
            release_after_first=asyncio.Event(),
        )
        queue = RecordingSegmentQueue()
        published = []
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=queue,
            generation_clock=clock,
            cancellation_token=cancellation,
            segmenter=LanguageAwareTextSegmenter(first_min_chars=2),
            on_token=published.append,
        )
        task = asyncio.create_task(pipeline.run("问题"))
        await first_token_seen.wait()
        cancellation.cancel()
        llm.release_after_first.set()

        result = await task

        self.assertTrue(result.cancelled)
        self.assertEqual([token.text for token in published], ["第一句。"])
        self.assertEqual([segment.text for segment in queue.items], ["第一句。"])
        self.assertEqual(llm.cancelled, 1)

    async def test_response_ids_are_unique_and_segment_ids_restart_for_each_serialized_run(self):
        clock = GenerationClock()
        llm = ControlledTokenProvider(("甲。",))
        queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=queue,
            generation_clock=clock,
            segmenter=LanguageAwareTextSegmenter(first_min_chars=1),
        )

        first, second = await asyncio.gather(pipeline.run("a"), pipeline.run("b"))

        self.assertNotEqual(first.response_id, second.response_id)
        segments = [item for item in queue.items if isinstance(item, TextSegment)]
        self.assertEqual([item.segment_id for item in segments], [0, 0])
        self.assertEqual([item.response_id for item in segments], [first.response_id, second.response_id])

    async def test_cancelling_run_cancels_provider_and_leaves_no_producer_work(self):
        clock = GenerationClock()
        first_token_seen = asyncio.Event()
        blocked = asyncio.Event()
        llm = ControlledTokenProvider(("等待", "后续"), first_token_seen=first_token_seen, release_after_first=blocked)
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=RecordingSegmentQueue(),
            generation_clock=clock,
        )
        task = asyncio.create_task(pipeline.run("问题"))
        await first_token_seen.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(llm.cancelled, 1)
        self.assertEqual(llm.closed, 1)

    async def test_provider_failure_is_propagated_after_provider_cleanup(self):
        pipeline = RealtimeResponsePipeline(
            llm=ControlledTokenProvider(("开始", "失败"), failure=1),
            segment_queue=RecordingSegmentQueue(),
            generation_clock=GenerationClock(),
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await pipeline.run("问题")

    async def test_pipeline_resets_its_own_cancelled_provider_before_a_later_valid_run(self):
        clock = GenerationClock()
        llm = ResetAfterCancelProvider()
        queue = RecordingSegmentQueue()
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=queue,
            generation_clock=clock,
            segmenter=LanguageAwareTextSegmenter(first_min_chars=1),
        )
        first = asyncio.create_task(pipeline.run("旧问题"))
        await llm.first_seen.wait()
        clock.advance()
        llm.release.set()
        stale = await first

        fresh = await pipeline.run("新问题")

        self.assertTrue(stale.stale)
        self.assertEqual(fresh.text, "新回答。")
        self.assertEqual(llm.reset_calls, 1)
        self.assertEqual(
            [item.text for item in queue.items if isinstance(item, TextSegment)],
            ["新回答。"],
        )

    async def test_externally_cancelled_token_never_resets_the_provider(self):
        cancellation = CancellationToken()
        cancellation.cancel()
        llm = ResetProbeProvider()
        pipeline = RealtimeResponsePipeline(
            llm=llm,
            segment_queue=RecordingSegmentQueue(),
            generation_clock=GenerationClock(),
            cancellation_token=cancellation,
        )

        first = await pipeline.run("问题")
        second = await pipeline.run("问题")

        self.assertTrue(first.cancelled)
        self.assertTrue(second.cancelled)
        self.assertEqual(llm.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
