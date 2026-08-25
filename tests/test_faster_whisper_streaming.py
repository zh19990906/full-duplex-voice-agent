import asyncio
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from src.adapters.asr.providers.faster_whisper_streaming import (
    FasterWhisperStreamingProvider,
    _stitch_shifted_hypothesis,
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


class LazySegments:
    def __init__(self, text):
        self._text = text
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        time.sleep(0.05)
        return {"text": self._text}


class LazySegmentsRuntime:
    def transcribe(self, pcm, **options):
        return LazySegments("lazy"), {"language": "en"}


def _apply_revision(chunks):
    committed = ""
    display = ""
    for chunk in chunks:
        if chunk.replaces_committed:
            committed = chunk.committed_text
        else:
            committed += chunk.text
        display = committed + chunk.unstable_text
    return display


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

    async def test_sync_lazy_segment_iteration_is_off_event_loop(self):
        provider = FasterWhisperStreamingProvider("model", runtime=LazySegmentsRuntime())
        ticks = 0
        running = True

        async def tick():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        await provider.push_pcm(PCM)
        result = await provider.finalize_turn()
        running = False
        await ticker
        self.assertGreaterEqual(ticks, 5)
        self.assertEqual(result.text, "lazy")

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

    async def test_final_major_correction_replaces_committed_text(self):
        runtime = SyncRuntime(["old direction", "old direction text", "new request"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        chunks = []
        for _ in range(10):
            result = await provider.push_pcm(PCM)
            if result is not None:
                chunks.append(result)
        for _ in range(10):
            result = await provider.push_pcm(PCM)
            if result is not None:
                chunks.append(result)
        chunks.append(await provider.finalize_turn())

        self.assertTrue(chunks[-1].replaces_committed)
        self.assertEqual(chunks[-1].committed_text, "new request")
        self.assertEqual(_apply_revision(chunks), "new request")
        self.assertTrue(chunks[-1].to_dict()["replaces_committed"])

    async def test_final_shrink_replaces_committed_text(self):
        runtime = SyncRuntime(["abcdef", "abcdefghi", "abc"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        chunks = []
        for _ in range(20):
            result = await provider.push_pcm(PCM)
            if result is not None:
                chunks.append(result)
        chunks.append(await provider.finalize_turn())

        self.assertTrue(chunks[-1].replaces_committed)
        self.assertEqual(_apply_revision(chunks), "abc")

    async def test_publication_identity_is_monotonic_across_turns_and_reset(self):
        runtime = SyncRuntime(["one", "one", "two"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        for _ in range(10):
            first = await provider.push_pcm(PCM)
        first_final = await provider.finalize_turn()
        await provider.push_pcm(PCM)
        await provider.cancel()
        provider.reset()
        await provider.push_pcm(PCM)
        second_final = await provider.finalize_turn()

        self.assertEqual([first.revision_id, first_final.revision_id, second_final.revision_id], [1, 2, 3])
        self.assertEqual(len({first.chunk_id, first_final.chunk_id, second_final.chunk_id}), 3)

    async def test_shifted_rolling_window_stitches_global_hypothesis(self):
        runtime = SyncRuntime(["abcd", "cdef", "efgh"])
        provider = FasterWhisperStreamingProvider(
            "model", runtime=runtime, cadence_ms=200, context_seconds=0.04
        )
        chunks = []
        for _ in range(20):
            result = await provider.push_pcm(PCM)
            if result is not None:
                chunks.append(result)
        # The final decode follows an actual trim, so it may align its local
        # window hypothesis with the preceding global transcript.
        await provider.push_pcm(PCM)
        chunks.append(await provider.finalize_turn())

        self.assertFalse(chunks[-1].replaces_committed)
        self.assertEqual(_apply_revision(chunks), "abcdefgh")
        self.assertEqual("".join(chunk.text for chunk in chunks), "abcdefgh")

    def test_shifted_alignment_prefers_latest_longest_anchor(self):
        self.assertEqual(_stitch_shifted_hypothesis("abcd", "cdef"), "abcdef")
        self.assertEqual(
            _stitch_shifted_hypothesis("turn left at", "left on red"),
            "turn left on red",
        )
        self.assertEqual(
            _stitch_shifted_hypothesis("go left then left at", "left on red"),
            "go left then left on red",
        )
        self.assertEqual(_stitch_shifted_hypothesis("alpha", "beta"), "beta")

    async def test_unshifted_correction_does_not_use_window_stitching(self):
        runtime = SyncRuntime(["turn left at", "turn right"])
        provider = FasterWhisperStreamingProvider("model", runtime=runtime, cadence_ms=200)
        for _ in range(10):
            first = await provider.push_pcm(PCM)
        for _ in range(10):
            second = await provider.push_pcm(PCM)

        self.assertEqual(first.unstable_text, "turn left at")
        self.assertEqual(second.text, "turn ")
        self.assertEqual(second.unstable_text, "right")

    async def test_lifetime_decode_metrics_include_source_and_end_to_end_rtf(self):
        provider = FasterWhisperStreamingProvider(
            "model", runtime=SyncRuntime(["metric"]), cadence_ms=200
        )
        for _ in range(10):
            await provider.push_pcm(PCM)
        self.assertAlmostEqual(provider.source_audio_seconds, 0.2)
        self.assertEqual(provider.total_decode_seconds, sum(provider.decode_durations))
        self.assertAlmostEqual(
            provider.end_to_end_decode_rtf,
            provider.total_decode_seconds / provider.source_audio_seconds,
        )

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

    def test_benchmark_absolute_script_path_bootstraps_project_from_outside_cwd(self):
        script = Path(__file__).parents[1] / "scripts" / "benchmark_streaming_asr.py"
        with tempfile.TemporaryDirectory() as outside_cwd:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--model",
                    "/definitely/missing/model",
                    "--audio",
                    "/definitely/missing/audio.wav",
                ],
                cwd=outside_cwd,
                capture_output=True,
                check=False,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model directory does not exist", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
