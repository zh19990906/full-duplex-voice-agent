import asyncio
import base64
import json
import unittest

from src.adapters.tts.providers.cosyvoice_worker import (
    CosyVoiceWorkerClient,
    decode_worker_message,
    encode_worker_request,
)


class CosyVoiceWorkerProtocolTests(unittest.TestCase):
    def test_encode_request_is_jsonl_and_preserves_generation_options(self):
        encoded = encode_worker_request(
            "req-1",
            "hello",
            prompt_text="reference",
            prompt_audio="/tmp/reference.wav",
        )

        message = json.loads(encoded)
        self.assertEqual(message["op"], "synthesize")
        self.assertEqual(message["request_id"], "req-1")
        self.assertEqual(message["text"], "hello")
        self.assertEqual(message["prompt_text"], "reference")
        self.assertTrue(encoded.endswith("\n"))

    def test_decode_chunk_message_returns_pcm_audio_chunk(self):
        payload = base64.b64encode(b"pcm").decode("ascii")
        chunk = decode_worker_message(
            json.dumps(
                {
                    "type": "chunk",
                    "request_id": "req-1",
                    "chunk_id": "chunk-1",
                    "audio_b64": payload,
                    "timestamp": 2.5,
                    "is_final": False,
                }
            )
        )

        self.assertEqual(chunk.chunk_id, "chunk-1")
        self.assertEqual(chunk.audio_data, b"pcm")
        self.assertEqual(chunk.timestamp, 2.5)
        self.assertFalse(chunk.is_final)

    def test_client_forwards_worker_command_and_normalizes_stream(self):
        class FakeWorker:
            async def stream_audio(self, text, **options):
                self.text = text
                self.options = options

                async def chunks():
                    yield {"audio_data": b"part", "is_final": False}
                    yield {"audio_data": b"done", "is_final": True}

                return chunks()

            async def interrupt(self):
                self.interrupted = True

        async def run():
            worker = FakeWorker()
            client = CosyVoiceWorkerClient(
                "/models/cosyvoice",
                prompt_audio="/tmp/prompt.wav",
                prompt_text="prompt",
                runtime=worker,
            )
            result = await client.stream_audio("hello")
            return [item async for item in result], worker

        chunks, worker = asyncio.run(run())
        self.assertEqual([chunk.audio_data for chunk in chunks], [b"part", b"done"])
        self.assertEqual(worker.text, "hello")
        self.assertEqual(worker.options["prompt_audio"], "/tmp/prompt.wav")
        self.assertEqual(worker.options["prompt_text"], "prompt")


if __name__ == "__main__":
    unittest.main()
