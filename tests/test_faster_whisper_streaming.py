import asyncio
from pathlib import Path
import subprocess
import sys
import time
import unittest

from src.adapters.asr.providers.faster_whisper_streaming import (
    FasterWhisperStreamingProvider,
)
from src.asr.stream import TranscriptChunk
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import AudioFrameHeader, PCM16_FRAME_BYTES


PCM = b"\x00\x00" * (PCM16_FRAME_BYTES // 2)


class SyncRuntime:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def transcribe(self, pcm, **options):
        self.calls.append((bytes(pcm), dict(options)))
        return self.results.pop(0)


class SlowSyncRuntime(SyncRuntime):
    def transcribe(self, pcm, **options):
        time.sleep(0.05)
        return super().transcribe(pcm, **options)


class AsyncRuntime(SyncRuntime):
    async def transcribe(self, pcm, **options):
        await asyncio.sleep(0)
        return super().transcribe(pcm, **options)


class GateAsyncRuntime:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, pcm, **options):
        self.started.set()
        await self.release.wait()
        return "late result"


class FasterWhisperStreamingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_cadence_rolls_context_and_finalizes_incrementally(self):
        runtime = SyncRuntime([
            ([{"text": "我想去北"}], {"language": "zh"}),
            ([{"text": "我想去北京"}], {}),
            ([{"text": "我想去北京旅游"}], {}),
        ])
        provider = FasterWhisperStreamingProvider(
            "models/asr", runtime=runtime, cadence_ms=200, context_seconds=0.04
        )

        for _ in range(9):
            self.assertIsNone(await provider.push_pcm(PCM))
        partial = await provider.push_pcm(PCM)
        self.assertIsNotNone(partial)
        self.assertEqual(partial.text, "")
        self.assertEqual(partial.unstable_text, "我想去北")
        self.assertEqual(partial.revision_id, 1)

        for _ in range(9):
            self.assertIsNone(await provider.push_pcm(PCM))
        partial = await provider.push_pcm(PCM)
        self.assertEqual(partial.text, "我想去北")
        self.assertEqual(partial.unstable_text, "京")
        self.assertEqual(partial.revision_id, 2)
        self.assertEqual([len(call[0]) for call in runtime.calls], [
            PCM16_FRAME_BYTES * 2,
            PCM16_FRAME_BYTES * 2,
        ])

        final = await provider.finalize_turn()
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "京旅游")
        self.assertEqual(final.unstable_text, "")
        self.assertEqual(final.revision_id, 3)
        self.assertEqual("".join(["我想去北", final.text]), "我想去北京旅游")

    async def test_boundaries_frame_metadata_and_result_shapes(self):
        for cadence in (199, 401):
            with self.assertRaises(ValueError):
                FasterWhisperStreamingProvider("model", runtime=SyncRuntime([]), cadence_ms=cadence)
        provider = FasterWhisperStreamingProvider(
            "model", runtime=SyncRuntime(["hello"]), cadence_ms=400
        )
        frame = RealtimeAudioFrame(AudioFrameHeader(0, 0.0), PCM)
        self.assertIsNone(await provider.push_pcm(frame))
        for _ in range(18):
            self.assertIsNone(await provider.push_pcm(PCM))
        self.assertIsNotNone(await provider.push_pcm(PCM))
        with self.assertRaises(ValueError):
            await provider.push_pcm(RealtimeAudioFrame(AudioFrameHeader(21, 0.0, 8000), PCM))

    async def test_exact_cadence_boundaries_and_final_decode_below_cadence(self):
        for cadence_ms, frame_count in ((200, 10), (400, 20)):
            runtime = SyncRuntime(["stable"])
            provider = FasterWhisperStreamingProvider(
                "model", runtime=runtime, cadence_ms=cadence_ms
            )
            for _ in range(frame_count - 1):
                self.assertIsNone(await provider.push_pcm(PCM))
            self.assertIsNotNone(await provider.push_pcm(PCM))

        runtime = SyncRuntime(["final only"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime)
        self.assertIsNone(await provider.push_pcm(PCM))
        final = await provider.finalize_turn()
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "final only")
        self.assertEqual(len(runtime.calls), 1)

    async def test_sync_decode_is_off_event_loop_and_tuple_segments_normalize(self):
        runtime = SlowSyncRuntime([([type("Segment", (), {"text": "hello"})()], {})])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime)
        ticks = 0

        async def tick():
            nonlocal ticks
            while ticks < 3:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        await provider.push_pcm(PCM)
        result = await provider.finalize_turn()
        await ticker
        self.assertGreaterEqual(ticks, 3)
        self.assertEqual(result.text, "hello")

    async def test_async_runtime_serializes_concurrent_push_finalize_and_cancel(self):
        runtime = AsyncRuntime(["a", "ab", "abc"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        for _ in range(9):
            await provider.push_pcm(PCM)
        first, final = await asyncio.gather(provider.push_pcm(PCM), provider.finalize_turn())
        self.assertIsNotNone(first)
        self.assertTrue(final.is_final)
        self.assertLess(first.revision_id, final.revision_id)

        await provider.push_pcm(PCM)
        await provider.cancel()
        with self.assertRaises(RuntimeError):
            await provider.push_pcm(PCM)
        provider.reset()
        self.assertEqual((await provider.finalize_turn()).text, "")

    async def test_cancel_suppresses_late_decode_publication(self):
        runtime = GateAsyncRuntime()
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        for _ in range(9):
            await provider.push_pcm(PCM)
        pending_push = asyncio.create_task(provider.push_pcm(PCM))
        await runtime.started.wait()
        cancelling = asyncio.create_task(provider.cancel())
        await asyncio.sleep(0)
        runtime.release.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            await pending_push
        await cancelling
        self.assertEqual(provider.context_bytes, b"")

    def test_transcript_chunk_keeps_positional_contract_and_validates_revision(self):
        chunk = TranscriptChunk("id", "text", 1.0, False)
        self.assertEqual(chunk.revision_id, 0)
        self.assertEqual(chunk.unstable_text, "")
        self.assertEqual(chunk.to_dict()["revision_id"], 0)
        with self.assertRaises(ValueError):
            TranscriptChunk("id", "text", 1.0, False, -1)

    def test_benchmark_help_is_model_free(self):
        script = Path(__file__).parents[1] / "scripts" / "benchmark_streaming_asr.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local faster-whisper model directory", result.stdout)


if __name__ == "__main__":
    unittest.main()
