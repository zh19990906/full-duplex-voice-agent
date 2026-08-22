import asyncio
import unittest

from src.audio.buffer import AudioBuffer
from src.audio.frames import AudioFrame
from src.audio.stream import AudioStream


class AudioStreamTest(unittest.IsolatedAsyncioTestCase):
    def test_audio_frame_creation_and_serialization(self):
        frame = AudioFrame(
            frame_id="001",
            timestamp=1.5,
            sample_rate=16000,
            channels=1,
            data=b"audio",
        )

        self.assertEqual(frame.to_dict(), {
            "frame_id": "001",
            "timestamp": 1.5,
            "sample_rate": 16000,
            "channels": 1,
            "data": b"audio",
        })

    async def test_buffer_is_fifo(self):
        buffer = AudioBuffer()
        first = self._frame("first")
        second = self._frame("second")

        await buffer.push(first)
        await buffer.push(second)

        self.assertIs(await buffer.pop(), first)
        self.assertIs(await buffer.pop(), second)

    async def test_buffer_pop_waits_for_async_push(self):
        buffer = AudioBuffer()
        pending_pop = asyncio.create_task(buffer.pop())
        await asyncio.sleep(0)
        self.assertFalse(pending_pop.done())

        frame = self._frame("later")
        await buffer.push(frame)

        self.assertIs(await pending_pop, frame)

    def test_stream_interface_imports_with_async_contract(self):
        self.assertTrue(issubclass(AudioStream, object))
        self.assertTrue(getattr(AudioStream.read, "__isabstractmethod__", False))
        self.assertTrue(getattr(AudioStream.write, "__isabstractmethod__", False))

    @staticmethod
    def _frame(frame_id):
        return AudioFrame(frame_id, 0.0, 16000, 1, b"x")


if __name__ == "__main__":
    unittest.main()
