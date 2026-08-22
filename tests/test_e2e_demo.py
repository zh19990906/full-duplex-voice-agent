import asyncio
import unittest

from src.adapters.asr.base import BaseASRAdapter
from src.adapters.audio.microphone import MicrophoneAdapter
from src.adapters.audio.speaker import SpeakerAdapter
from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.tts.base import BaseTTSAdapter
from src.audio.frames import AudioFrame
from src.demo.voice_agent_demo import VoiceAgentDemo, load_demo_config, run_demo
from src.llm_runtime.stream import TokenChunk


class FakeMicBackend:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def read(self):
        return await self.queue.get()

    async def stop(self):
        self.stopped = True


class FakeSpeakerBackend:
    def __init__(self):
        self.frames = []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def write(self, frame):
        self.frames.append(frame)

    async def stop(self):
        self.stopped = True


class FakeASR(BaseASRAdapter):
    async def stream_audio(self, audio_chunk):
        return {"text": audio_chunk.decode(), "is_final": True}


class FakeLLM(BaseLLMAdapter):
    async def generate(self, prompt):
        return "reply"

    async def stream_tokens(self, prompt):
        async def stream():
            yield TokenChunk("one", "reply", 1.0, False)
            yield TokenChunk("final", "", 2.0, True)
        return stream()

    async def cancel(self):
        pass


class FakeTTS(BaseTTSAdapter):
    async def synthesize(self, text):
        pass

    async def stream_audio(self, text):
        return [text.encode()]

    async def interrupt(self):
        pass


class EndToEndDemoTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_and_audio_pipeline(self):
        mic_backend = FakeMicBackend()
        speaker_backend = FakeSpeakerBackend()
        demo = VoiceAgentDemo(
            MicrophoneAdapter(mic_backend),
            SpeakerAdapter(speaker_backend),
            FakeASR(),
            FakeLLM(),
            FakeTTS(),
        )
        await demo.start()
        await mic_backend.queue.put(b"hello")
        for _ in range(30):
            if speaker_backend.frames:
                break
            await asyncio.sleep(0)
        self.assertTrue(demo.application.initialized)
        self.assertTrue(mic_backend.started)
        self.assertTrue(speaker_backend.frames)
        await demo.stop()
        self.assertTrue(speaker_backend.stopped)

    async def test_existing_backchannel_and_interrupt_scenarios_remain_available(self):
        results = await run_demo()
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.passed for result in results), results)

    def test_demo_configuration(self):
        config = load_demo_config()
        self.assertEqual(config["audio"]["input_provider"], "sounddevice")
        self.assertEqual(config["models"]["llm"], "llama_cpp")


if __name__ == "__main__":
    unittest.main()
