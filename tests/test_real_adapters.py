import asyncio
import unittest

from src.adapters.asr.backend import StreamingASRBackend
from src.adapters.llm.backend import StreamingLLMBackend
from src.adapters.tts.backend import StreamingTTSBackend


class ASRProvider:
    async def stream_audio(self, audio):
        return {"text": audio.decode("utf-8"), "is_final": False}


class LLMProvider:
    def __init__(self):
        self.cancelled = False

    async def stream_tokens(self, prompt):
        async def stream():
            yield "one"
            if not self.cancelled:
                yield "two"

        return stream()

    async def cancel(self):
        self.cancelled = True


class TTSProvider:
    def __init__(self):
        self.interrupted = False

    async def stream_audio(self, text):
        return [] if self.interrupted else [text.encode("utf-8")]

    async def interrupt(self):
        self.interrupted = True


class RealAdapterTests(unittest.TestCase):
    def test_asr_streaming_and_model_path(self):
        async def run():
            adapter = StreamingASRBackend(ASRProvider(), model_path="models/asr")
            return await adapter.stream_audio(b"partial")

        result = asyncio.run(run())
        self.assertEqual(result["text"], "partial")

    def test_llm_streaming_and_cancellation(self):
        async def run():
            provider = LLMProvider()
            adapter = StreamingLLMBackend(provider, model_path="models/llm")
            stream = await adapter.stream_tokens("prompt")
            first = [item async for item in stream]
            await adapter.cancel()
            return first, provider.cancelled

        result, cancelled = asyncio.run(run())
        self.assertEqual(result, ["one", "two"])
        self.assertTrue(cancelled)

    def test_tts_streaming_and_interrupt(self):
        async def run():
            provider = TTSProvider()
            adapter = StreamingTTSBackend(provider, model_path="models/tts")
            before = await adapter.stream_audio("hello")
            await adapter.interrupt()
            try:
                await adapter.stream_audio("hello")
            except RuntimeError:
                interrupted_output_rejected = True
            else:
                interrupted_output_rejected = False
            return before, interrupted_output_rejected, provider.interrupted

        before, output_rejected, interrupted = asyncio.run(run())
        self.assertEqual(before, [b"hello"])
        self.assertTrue(output_rejected)
        self.assertTrue(interrupted)


if __name__ == "__main__":
    unittest.main()
