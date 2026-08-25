import asyncio
import unittest

from src.llm_runtime.stream import TokenChunk
from src.realtime.text_segmenter import LanguageAwareTextSegmenter
from src.runtime_app.real_session import RealModelSession
from src.tts_runtime.stream import AudioChunk


class FakeLLM:
    async def stream_tokens(self, prompt):
        self.prompt = prompt
        yield TokenChunk("t1", "Hello", 1.0, False)
        yield TokenChunk("t2", " there", 2.0, True)

    async def cancel(self):
        self.cancelled = True


class FakeTTS:
    async def stream_audio(self, text):
        self.text = text

        async def chunks():
            yield AudioChunk("a1", b"pcm", 3.0, False)
            yield AudioChunk("a2", b"", 4.0, True)

        return chunks()

    async def interrupt(self):
        self.interrupted = True


class SegmentedTTS:
    def __init__(self):
        self.calls = []

    async def stream_audio(self, text):
        self.calls.append(text)

        async def chunks():
            yield AudioChunk(f"audio-{len(self.calls)}", text.encode(), 3.0, True)

        return chunks()


class BlockingTTS:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.interrupted = False

    async def stream_audio(self, text):
        async def chunks():
            self.entered.set()
            await self.release.wait()
            yield AudioChunk("late", b"late", 3.0, True)

        return chunks()

    async def interrupt(self):
        self.interrupted = True


class FailingTTS:
    async def stream_audio(self, text):
        raise RuntimeError("tts failed")

    async def interrupt(self):
        return None


class ResetRequiredLLM(FakeLLM):
    def __init__(self):
        self.cancelled = False
        self.reset_calls = 0

    async def stream_tokens(self, prompt):
        if self.cancelled:
            raise RuntimeError("provider remains cancelled")
        async for token in super().stream_tokens(prompt):
            yield token

    async def cancel(self):
        self.cancelled = True

    def reset(self):
        self.reset_calls += 1
        self.cancelled = False


class RealModelSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_publishes_tokens_and_audio_in_order(self):
        events = []

        async def publish(value):
            events.append(value)

        llm = FakeLLM()
        tts = FakeTTS()
        session = RealModelSession(llm, tts, publish, prompt_builder=lambda text: f"prompt:{text}")

        response = await session.run("question")

        self.assertEqual(response, "Hello there")
        self.assertEqual([type(event) for event in events], [TokenChunk, TokenChunk, AudioChunk, AudioChunk])
        self.assertEqual(llm.prompt, "prompt:question")
        self.assertEqual(tts.text, "Hello there")

    async def test_interrupt_forwards_to_both_model_providers(self):
        llm = FakeLLM()
        tts = FakeTTS()
        session = RealModelSession(llm, tts, lambda _value: None)

        await session.interrupt()

        self.assertTrue(llm.cancelled)
        self.assertTrue(tts.interrupted)

    async def test_run_synthesizes_stable_segments_in_order_without_full_response_duplicate(self):
        events = []
        tts = SegmentedTTS()

        session = RealModelSession(
            FakeLLM(),
            tts,
            events.append,
            segmenter=LanguageAwareTextSegmenter(first_min_words=1, next_min_words=1),
        )

        response = await session.run("question")

        self.assertEqual(response, "Hello there")
        self.assertEqual(tts.calls, ["Hello", " there"])
        self.assertEqual(
            [event.audio_data for event in events if isinstance(event, AudioChunk)],
            [b"Hello", b" there"],
        )

    async def test_interrupt_invalidates_active_epoch_and_discards_late_tts_audio(self):
        events = []
        tts = BlockingTTS()
        session = RealModelSession(
            FakeLLM(),
            tts,
            events.append,
            segmenter=LanguageAwareTextSegmenter(first_min_words=1, next_min_words=1),
        )

        task = asyncio.create_task(session.run("question"))
        await tts.entered.wait()
        active_epoch = session.generation_clock.current
        await session.interrupt()
        tts.release.set()
        await task

        self.assertEqual(session.generation_clock.current, active_epoch + 1)
        self.assertTrue(tts.interrupted)
        self.assertEqual([event for event in events if isinstance(event, AudioChunk)], [])

    async def test_tts_consumer_failure_cancels_generation_and_propagates_without_deadlock(self):
        llm = FakeLLM()
        session = RealModelSession(
            llm,
            FailingTTS(),
            lambda _value: None,
            segmenter=LanguageAwareTextSegmenter(first_min_words=1, next_min_words=1),
        )

        with self.assertRaisesRegex(RuntimeError, "tts failed"):
            await asyncio.wait_for(session.run("question"), timeout=1.0)

        self.assertTrue(llm.cancelled)

    async def test_new_run_resets_a_provider_cancelled_by_a_previous_interrupt(self):
        llm = ResetRequiredLLM()
        session = RealModelSession(llm, FakeTTS(), lambda _value: None)

        await session.interrupt()
        response = await session.run("question")

        self.assertEqual(response, "Hello there")
        self.assertEqual(llm.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
