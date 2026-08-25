import asyncio
import base64
import json
import sys
import unittest
from unittest.mock import ANY

from src.adapters.tts.providers.cosyvoice_worker import (
    CosyVoiceWorkerClient,
    decode_worker_message,
    encode_worker_cancel,
    encode_worker_request,
)
from src.tts_runtime.stream import AudioChunk


async def collect_with_timeout(stream, timeout=0.5):
    return await asyncio.wait_for(_collect(stream), timeout=timeout)


async def _collect(stream):
    return [item async for item in stream]


class ScriptedWorkerTransport:
    """In-memory JSONL worker used to exercise the client protocol."""

    def __init__(self, messages=()):
        self.messages = asyncio.Queue()
        for message in messages:
            self.messages.put_nowait(json.dumps(message) + "\n")
        self.sent = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def send(self, line):
        self.sent.append(json.loads(line))

    async def readline(self):
        return await self.messages.get()

    async def close(self):
        self.closed = True


class CosyVoiceWorkerProtocolTests(unittest.TestCase):
    def test_worker_cancel_message_contains_request_identity(self):
        self.assertEqual(
            json.loads(encode_worker_cancel("request-7")),
            {"op": "cancel", "request_id": "request-7"},
        )

    def test_worker_request_identity_must_be_nonempty(self):
        with self.assertRaises(ValueError):
            encode_worker_request("", "hello")
        with self.assertRaises(ValueError):
            encode_worker_cancel("")
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

    def test_decode_chunk_preserves_realtime_identity(self):
        payload = base64.b64encode(b"pcm").decode("ascii")
        chunk = decode_worker_message(
            json.dumps(
                {
                    "event": "audio",
                    "request_id": "req-1",
                    "pcm": payload,
                    "response_id": "response-1",
                    "generation_epoch": 2,
                    "segment_id": 3,
                }
            )
        )

        self.assertEqual(chunk.request_id, "req-1")
        self.assertEqual(chunk.response_id, "response-1")
        self.assertEqual(chunk.generation_epoch, 2)
        self.assertEqual(chunk.segment_id, 3)

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
            self.assertEqual(client.startup_timeout, 120.0)
            result = await client.stream_audio("hello")
            return [item async for item in result], worker

        chunks, worker = asyncio.run(run())
        self.assertEqual([chunk.audio_data for chunk in chunks], [b"part", b"done"])
        self.assertEqual(worker.text, "hello")
        self.assertEqual(worker.options["prompt_audio"], "/tmp/prompt.wav")
        self.assertEqual(worker.options["prompt_text"], "prompt")


class CosyVoiceWorkerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_request_discards_late_audio_and_drains_terminal(self):
        transport = ScriptedWorkerTransport(
            messages=[
                {"event": "audio", "request_id": "request-7", "pcm": "AAAA"},
                {"event": "done", "request_id": "request-7"},
            ]
        )
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)

        stream = await client.stream_audio("文本", request_id="request-7")
        await client.cancel("request-7")

        self.assertEqual(await collect_with_timeout(stream), [])
        self.assertEqual(
            transport.sent,
            [
                {"op": "synthesize", "request_id": "request-7", "text": "文本"},
                {"op": "cancel", "request_id": "request-7"},
            ],
        )

    async def test_cancel_after_first_chunk_never_yields_late_audio(self):
        transport = ScriptedWorkerTransport(
            messages=[
                {"event": "audio", "request_id": "request-7", "pcm": "AAAA"},
                {"event": "audio", "request_id": "request-7", "pcm": "AQID"},
                {"event": "cancelled", "request_id": "request-7"},
            ]
        )
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)
        stream = await client.stream_audio("文本", request_id="request-7")

        first = await asyncio.wait_for(anext(stream), timeout=0.5)
        await client.cancel("request-7")

        self.assertEqual(first.audio_data, b"\x00\x00\x00")
        self.assertEqual(await collect_with_timeout(stream), [])

    async def test_malformed_audio_marks_transport_broken_without_yielding(self):
        transport = ScriptedWorkerTransport(
            messages=[
                {"event": "audio", "request_id": "request-7", "pcm": "not base64!"},
            ]
        )
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)
        stream = await client.stream_audio("文本", request_id="request-7")

        with self.assertRaisesRegex(RuntimeError, "malformed audio"):
            await asyncio.wait_for(anext(stream), timeout=0.5)
        with self.assertRaisesRegex(RuntimeError, "transport is broken"):
            await client.stream_audio("next", request_id="request-8")
        self.assertTrue(transport.closed)

    async def test_runtime_receives_request_identity_and_cancel_identity(self):
        class Runtime:
            def __init__(self):
                self.calls = []
                self.cancelled = []

            async def stream_audio(self, text, **options):
                self.calls.append((text, options))

                async def chunks():
                    yield b"pcm"

                return chunks()

            async def cancel(self, request_id):
                self.cancelled.append(request_id)

        runtime = Runtime()
        client = CosyVoiceWorkerClient("/models/cosyvoice", runtime=runtime)
        stream = await client.stream_audio(
            "文本", request_id="request-7", response_id="response-1", generation_epoch=4, segment_id=2
        )
        await client.cancel("request-7")

        self.assertEqual(await collect_with_timeout(stream), [])
        self.assertEqual(runtime.calls[0][1]["request_id"], "request-7")
        self.assertEqual(runtime.cancelled, ["request-7"])

    async def test_runtime_without_request_aware_cancel_uses_legacy_interrupt(self):
        class Runtime:
            def __init__(self):
                self.interrupted = 0

            async def stream_audio(self, text):
                async def chunks():
                    yield b"late"

                return chunks()

            async def interrupt(self):
                self.interrupted += 1

        runtime = Runtime()
        client = CosyVoiceWorkerClient("/models/cosyvoice", runtime=runtime)
        stream = await client.stream_audio("文本", request_id="request-7")

        await client.cancel("request-7")
        self.assertEqual(await collect_with_timeout(stream), [])
        self.assertEqual(runtime.interrupted, 1)

    async def test_subprocess_transport_starts_and_reads_the_single_active_request(self):
        worker = (
            "import json,sys; print(json.dumps({'type':'ready'}), flush=True); "
            "line=sys.stdin.readline(); request=json.loads(line); "
            "print(json.dumps({'type':'done','request_id':request['request_id']}), flush=True)"
        )
        client = CosyVoiceWorkerClient(
            "/models/cosyvoice", worker_command=[sys.executable, "-u", "-c", worker]
        )
        stream = await client.stream_audio("文本", request_id="request-7")

        self.assertEqual(await collect_with_timeout(stream), [])
        await client.close()

    async def test_subprocess_transport_keeps_identity_and_discards_late_audio_after_cancel(self):
        worker = "\n".join(
            [
                "import json, sys",
                "print(json.dumps({'type': 'ready'}), flush=True)",
                "request = json.loads(sys.stdin.readline())",
                "print(json.dumps({'type': 'chunk', 'request_id': request['request_id'], 'audio_b64': 'cGNt', 'is_final': False}), flush=True)",
                "cancel = json.loads(sys.stdin.readline())",
                "assert cancel == {'op': 'cancel', 'request_id': request['request_id']}",
                "print(json.dumps({'type': 'chunk', 'request_id': request['request_id'], 'audio_b64': 'bGF0ZQ==', 'is_final': False}), flush=True)",
                "print(json.dumps({'type': 'cancelled', 'request_id': request['request_id']}), flush=True)",
                "sys.stdin.readline()",
            ]
        )
        client = CosyVoiceWorkerClient(
            "/models/cosyvoice", worker_command=[sys.executable, "-u", "-c", worker]
        )
        stream = await client.stream_audio(
            "文本",
            request_id="request-7",
            response_id="response-4",
            generation_epoch=9,
            segment_id=2,
        )

        first = await asyncio.wait_for(anext(stream), timeout=0.5)
        await client.cancel("request-7")

        self.assertEqual(first.audio_data, b"pcm")
        self.assertEqual(
            (first.request_id, first.response_id, first.generation_epoch, first.segment_id),
            ("request-7", "response-4", 9, 2),
        )
        self.assertEqual(await collect_with_timeout(stream), [])
        await client.close()

    async def test_subprocess_eof_marks_transport_broken_after_request_start(self):
        worker = "\n".join(
            [
                "import json, sys",
                "print(json.dumps({'type': 'ready'}), flush=True)",
                "sys.stdin.readline()",
            ]
        )
        client = CosyVoiceWorkerClient(
            "/models/cosyvoice", worker_command=[sys.executable, "-u", "-c", worker]
        )
        stream = await client.stream_audio("文本", request_id="request-7")

        with self.assertRaisesRegex(RuntimeError, "protocol ended unexpectedly"):
            await asyncio.wait_for(anext(stream), timeout=0.5)
        with self.assertRaisesRegex(RuntimeError, "transport is broken"):
            await client.stream_audio("下一次", request_id="request-8")
        await client.close()

    async def test_client_propagates_every_identity_to_audio_chunks(self):
        transport = ScriptedWorkerTransport(
            messages=[
                {
                    "type": "chunk",
                    "request_id": "request-7",
                    "audio_b64": base64.b64encode(b"pcm").decode("ascii"),
                    "is_final": False,
                },
                {"type": "done", "request_id": "request-7"},
            ]
        )
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)

        stream = await client.stream_audio(
            "文本",
            request_id="request-7",
            response_id="response-4",
            generation_epoch=9,
            segment_id=2,
        )
        chunks = await collect_with_timeout(stream)

        self.assertEqual(
            chunks,
            [
                AudioChunk(
                    "cosyvoice-audio",
                    b"pcm",
                    ANY,
                    False,
                    "request-7",
                    "response-4",
                    9,
                    2,
                )
            ],
        )

    async def test_wrong_id_cancel_and_concurrent_streams_are_rejected(self):
        transport = ScriptedWorkerTransport()
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)
        await client.stream_audio("one", request_id="request-1")

        with self.assertRaisesRegex(ValueError, "does not match"):
            await client.cancel("request-2")
        with self.assertRaisesRegex(RuntimeError, "already active"):
            await client.stream_audio("two", request_id="request-2")

    async def test_next_request_runs_after_matching_terminal(self):
        transport = ScriptedWorkerTransport(
            messages=[
                {"event": "done", "request_id": "request-1"},
                {"event": "audio", "request_id": "request-2", "pcm": "AAAA"},
                {"event": "done", "request_id": "request-2"},
            ]
        )
        client = CosyVoiceWorkerClient("/models/cosyvoice", transport=transport)

        first = await client.stream_audio("one", request_id="request-1")
        self.assertEqual(await collect_with_timeout(first), [])
        second = await client.stream_audio("two", request_id="request-2")
        chunks = await collect_with_timeout(second)

        self.assertEqual([item.audio_data for item in chunks], [b"\x00\x00\x00"])


if __name__ == "__main__":
    unittest.main()
