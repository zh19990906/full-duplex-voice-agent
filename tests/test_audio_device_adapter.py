import asyncio
import tempfile
import unittest
from pathlib import Path

from src.adapters.audio.microphone import MicrophoneAdapter
from src.adapters.audio.speaker import SpeakerAdapter
from src.audio.factory import create_audio_input, create_audio_output, load_audio_config
from src.audio.frames import AudioFrame


class FakeMicrophoneBackend:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def read(self):
        if self.chunks:
            return self.chunks.pop(0)
        await asyncio.sleep(0)
        return b""

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


class AudioDeviceAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_microphone_converts_chunks_and_supports_async_iteration(self):
        backend = FakeMicrophoneBackend([b"one", b"two"])
        microphone = MicrophoneAdapter(backend, sample_rate=16000, channels=1)
        first = await microphone.read()
        second = await microphone.read()
        self.assertIsInstance(first, AudioFrame)
        self.assertEqual([first.data, second.data], [b"one", b"two"])
        self.assertTrue(backend.started)
        await microphone.stop()
        self.assertTrue(backend.stopped)

    async def test_microphone_stop_ends_async_iteration(self):
        backend = FakeMicrophoneBackend([b"one"])
        microphone = MicrophoneAdapter(backend)
        await microphone.stop()
        with self.assertRaises(RuntimeError):
            await microphone.read()

    async def test_speaker_preserves_frame_order_and_stops(self):
        backend = FakeSpeakerBackend()
        speaker = SpeakerAdapter(backend)
        frames = [AudioFrame(str(i), float(i), 16000, 1, bytes([i])) for i in range(3)]
        for frame in frames:
            await speaker.write(frame)
        self.assertEqual(backend.frames, frames)
        await speaker.stop()
        self.assertTrue(backend.stopped)
        with self.assertRaises(RuntimeError):
            await speaker.write(frames[0])

    async def test_factory_selects_adapters_from_config(self):
        backend_in = FakeMicrophoneBackend([])
        backend_out = FakeSpeakerBackend()
        config = {
            "input": {"provider": "fake", "sample_rate": 16000, "channels": 1},
            "output": {"provider": "fake", "sample_rate": 24000, "channels": 1},
        }
        microphone = create_audio_input(config, backend=backend_in)
        speaker = create_audio_output(config, backend=backend_out)
        self.assertIsInstance(microphone, MicrophoneAdapter)
        self.assertIsInstance(speaker, SpeakerAdapter)

    def test_audio_yaml_config_loads_without_external_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.yaml"
            path.write_text(
                "audio:\n"
                "  input:\n"
                "    provider: fake\n"
                "    sample_rate: 16000\n"
                "    channels: 1\n"
                "  output:\n"
                "    provider: fake\n"
                "    sample_rate: 24000\n"
                "    channels: 1\n",
                encoding="utf-8",
            )
            config = load_audio_config(path)
        self.assertEqual(config["input"]["sample_rate"], 16000)
        self.assertEqual(config["output"]["channels"], 1)


if __name__ == "__main__":
    unittest.main()
