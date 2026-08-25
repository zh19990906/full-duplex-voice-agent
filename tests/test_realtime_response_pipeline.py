import asyncio
import unittest

from src.llm_runtime.stream import TokenChunk
from src.realtime.cancellation import CancellationToken
from src.realtime.identifiers import GenerationClock
from src.realtime.response_pipeline import RealtimeResponsePipeline, ResponsePipelineResult
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
        self.assertEqual([item.text for item in queue.items], ["你好，", "世界。"])
        self.assertEqual([item.segment_id for item in queue.items], [0, 1])
        self.assertTrue(all(item.response_id == result.response_id for item in queue.items))

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
            [(segment.text, segment.is_final) for segment in queue.items],
            [("最后一句。", True)],
        )

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
        self.assertEqual([item.segment_id for item in queue.items], [0, 0])
        self.assertEqual([item.response_id for item in queue.items], [first.response_id, second.response_id])

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


if __name__ == "__main__":
    unittest.main()
