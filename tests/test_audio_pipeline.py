import asyncio
import unittest

from src.audio.buffer import AudioBuffer
from src.audio.frames import AudioFrame
from src.audio_pipeline.processor import AudioProcessor
from src.audio_pipeline.router import AudioRouter
from src.core.interfaces.turn import TurnAdapter


class FakeTurnAdapter(TurnAdapter):
    def __init__(self):
        self.received = []

    async def push_audio(self, audio_chunk):
        self.received.append(audio_chunk)


class FakeAudioStream:
    def __init__(self):
        self.frames = asyncio.Queue()

    async def read(self):
        return await self.frames.get()

    async def write(self, frame):
        await self.frames.put(frame)


class AudioPipelineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.turn_adapter = FakeTurnAdapter()
        self.router = AudioRouter(self.turn_adapter)
        self.buffer = AudioBuffer()
        self.stream = FakeAudioStream()
        self.processor = AudioProcessor(self.stream, self.buffer, self.router)

    async def test_audio_frame_can_enter_processor(self):
        await self.processor.process_frame(self._frame(b"audio"))

        self.assertEqual(self.turn_adapter.received, [b"audio"])

    async def test_router_forwards_original_bytes(self):
        data = b"original-bytes"

        await self.router.route(self._frame(data))

        self.assertIs(self.turn_adapter.received[0], data)

    async def test_multiple_frames_preserve_order(self):
        frames = [self._frame(b"one"), self._frame(b"two"), self._frame(b"three")]

        for frame in frames:
            await self.processor.process_frame(frame)

        self.assertEqual(self.turn_adapter.received, [b"one", b"two", b"three"])

    async def test_start_and_stop_lifecycle(self):
        await self.processor.start()
        self.assertTrue(self.processor.running)

        await self.stream.write(self._frame(b"streamed"))
        for _ in range(10):
            if self.turn_adapter.received:
                break
            await asyncio.sleep(0)

        await self.processor.stop()

        self.assertFalse(self.processor.running)
        self.assertEqual(self.turn_adapter.received, [b"streamed"])

    @staticmethod
    def _frame(data):
        return AudioFrame("frame", 0.0, 16000, 1, data)


if __name__ == "__main__":
    unittest.main()
