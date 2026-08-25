import asyncio
import unittest

from src.llm_runtime.stream import TokenChunk
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


if __name__ == "__main__":
    unittest.main()
