import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
import unittest

from src.adapters.turn.x2_turn_adapter import X2TurnAdapter
from src.adapters.turn.x2_turn_streaming import TurnCandidate, X2TurnRollingProvider
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import AudioFrameHeader, PCM16_FRAME_BYTES
from src.realtime.speech_fusion import SpeechEventFusion


PCM = b"\x00\x00" * (PCM16_FRAME_BYTES // 2)


class SequenceRuntime:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def infer(self, pcm):
        self.calls.append(bytes(pcm))
        return self.results.pop(0)


class SlowLazyRuntime:
    def infer(self, pcm):
        def labels():
            time.sleep(0.05)
            yield {"label": "speaking", "confidence": 0.75}

        return labels()


class GateRuntime:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer(self, pcm):
        self.started.set()
        await self.release.wait()
        return ["turn_end"]


@dataclass
class RuntimeResult:
    transcript: str
    turn_frames: list[object]


class X2TurnRollingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_cadence_boundaries_and_context_are_bounded(self):
        """Catches full inference per 20ms frame or an unbounded PCM window."""
        runtime = SequenceRuntime([["idle"]] * 11)
        provider = X2TurnRollingProvider(
            "models/x2", runtime=runtime, cadence_ms=100, context_seconds=1.0
        )

        for _ in range(4):
            self.assertEqual(await provider.push_pcm(PCM), ())
        self.assertEqual((await provider.push_pcm(PCM))[0].label, "idle")

        for _ in range(50):
            await provider.push_pcm(PCM)

        self.assertEqual(len(runtime.calls), 11)
        self.assertEqual(len(provider.context_bytes), 50 * PCM16_FRAME_BYTES)
        self.assertTrue(all(len(call) <= 50 * PCM16_FRAME_BYTES for call in runtime.calls))

    async def test_frame_identity_validation_and_latest_capture_timestamp_are_preserved(self):
        """Catches discarding ingress identity or accepting non-V1 audio frames."""
        runtime = SequenceRuntime([[{"label": "turn-end", "confidence": 0.8}]])
        provider = X2TurnRollingProvider("models/x2", runtime=runtime, cadence_ms=100)

        for sequence in range(4):
            self.assertEqual(
                await provider.push_pcm(
                    RealtimeAudioFrame(AudioFrameHeader(sequence, sequence / 10), PCM)
                ),
                (),
            )
        candidates = await provider.push_pcm(
            RealtimeAudioFrame(AudioFrameHeader(4, 9.25), PCM)
        )

        self.assertEqual(candidates[0].label, "turn_end")
        self.assertEqual(candidates[0].sequence, 4)
        self.assertEqual(candidates[0].capture_timestamp, 9.25)
        self.assertGreaterEqual(candidates[0].inference_duration, 0.0)
        self.assertGreaterEqual(candidates[0].rtf, 0.0)

        invalid_rate = RealtimeAudioFrame(AudioFrameHeader(5, 1.0, 8000), PCM)
        with self.assertRaisesRegex(ValueError, "16kHz"):
            await provider.push_pcm(invalid_rate)
        with self.assertRaisesRegex(ValueError, "exactly"):
            await provider.push_pcm(b"short")

    async def test_label_normalization_confidence_validation_and_x2_result_shapes(self):
        """Catches aliases or official wrapper result forms changing candidate semantics."""
        self.assertEqual(TurnCandidate("turn-end", 0.5).label, "turn_end")
        self.assertEqual(TurnCandidate("NO IDLE", 0.5).label, "noidle")
        self.assertEqual(TurnCandidate("back_channel", 0.5).label, "backchannel")
        with self.assertRaisesRegex(ValueError, "confidence"):
            TurnCandidate("idle", 1.1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            TurnCandidate("interrupt", 0.5)

        runtime = SequenceRuntime(
            [
                ("ignored by turn provider", ["speaking", "turn_end"]),
                {"turn_labels": {"idle": 2, "backchannel": 1}},
                RuntimeResult("also ignored", [{"label": "no-idle", "score": 0.6}]),
            ]
        )
        provider = X2TurnRollingProvider("models/x2", runtime=runtime, cadence_ms=100)

        first = await self._push_cadence(provider)
        second = await self._push_cadence(provider)
        third = await self._push_cadence(provider)

        self.assertEqual([item.label for item in first], ["speaking", "turn_end"])
        self.assertEqual([item.label for item in second], ["idle"])
        self.assertEqual(second[0].confidence, 2 / 3)
        self.assertEqual(second[0].metadata["counts"], {"idle": 2, "backchannel": 1})
        self.assertTrue(second[0].metadata["aggregated"])
        self.assertEqual(third[0].label, "noidle")
        self.assertEqual(third[0].confidence, 0.6)
        self.assertNotIn("transcript", third[0].metadata)

    async def test_label_counts_are_one_aggregate_window_not_a_fabricated_sequence(self):
        """Catches count mappings inventing speaking-to-turn-end chronology."""
        runtime = SequenceRuntime(
            [
                {"turn_labels": {"speaking": 3, "turn_end": 3}},
                {"turn_labels": {"turn_end": 2}},
                {"turn_labels": {"turn_end": 2}},
            ]
        )
        provider = X2TurnRollingProvider("models/x2", runtime=runtime, cadence_ms=100)
        fusion = SpeechEventFusion(turn_end_frames=2)

        mixed = await self._push_cadence(provider)
        self.assertEqual(len(mixed), 1)
        self.assertEqual(mixed[0].label, "speaking")
        self.assertEqual(mixed[0].confidence, 0.5)
        self.assertEqual(mixed[0].metadata, {
            "aggregated": True,
            "counts": {"speaking": 3, "turn_end": 3},
        })
        mixed_events = fusion.accept_turn(mixed[0])
        self.assertEqual([event.event for event in mixed_events], ["USER_SPEECH_START_CANDIDATE"])
        self.assertNotIn("USER_TURN_END_CANDIDATE", [event.event for event in mixed_events])

        self.assertEqual(fusion.accept_turn((await self._push_cadence(provider))[0]), ())
        self.assertEqual(
            [event.event for event in fusion.accept_turn((await self._push_cadence(provider))[0])],
            ["USER_TURN_END_CANDIDATE"],
        )

    async def test_sync_inference_and_lazy_result_normalization_do_not_block_event_loop(self):
        """Catches lazy official outputs being consumed on the realtime event loop."""
        provider = X2TurnRollingProvider("models/x2", runtime=SlowLazyRuntime(), cadence_ms=100)
        ticks = 0
        running = True

        async def tick():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        candidates = await self._push_cadence(provider)
        running = False
        await ticker

        self.assertGreaterEqual(ticks, 5)
        self.assertEqual(candidates[0].label, "speaking")

    async def test_concurrent_decode_cancel_and_reset_suppress_stale_publication(self):
        """Catches cancelled X2 output being published after a newer turn starts."""
        runtime = GateRuntime()
        provider = X2TurnRollingProvider("models/x2", runtime=runtime, cadence_ms=100)
        for _ in range(4):
            await provider.push_pcm(PCM)
        pending = asyncio.create_task(provider.push_pcm(PCM))
        await runtime.started.wait()
        cancelling = asyncio.create_task(provider.cancel())
        await asyncio.sleep(0)
        runtime.release.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            await pending
        await cancelling

        provider.reset()
        self.assertEqual(provider.context_bytes, b"")

    async def test_publication_identifiers_stay_monotonic_across_turn_reset(self):
        """Catches a reset reusing a candidate revision that downstream already saw."""
        runtime = SequenceRuntime([["idle"], ["speaking"]])
        provider = X2TurnRollingProvider("models/x2", runtime=runtime, cadence_ms=100)

        first = (await self._push_cadence(provider))[0]
        provider.reset()
        second = (await self._push_cadence(provider))[0]

        self.assertLess(first.revision_id, second.revision_id)

    async def _push_cadence(self, provider):
        for _ in range(4):
            self.assertEqual(await provider.push_pcm(PCM), ())
        return await provider.push_pcm(PCM)


class X2TurnAdapterCandidateCompatibilityTests(unittest.TestCase):
    def test_candidate_mapping_preserves_legacy_semantic_adapter_behavior(self):
        """Catches candidate mapping reusing legacy final User* event decisions."""
        adapter = X2TurnAdapter(backend=object())

        candidate = adapter.map_candidate(
            {"state": "turn-end", "confidence": 0.9, "timestamp": 3.5}
        )
        legacy = adapter.map_output({"state": "turn_end", "confidence": 0.9, "timestamp": 3.5})

        self.assertIsInstance(candidate, TurnCandidate)
        self.assertEqual(candidate.label, "turn_end")
        self.assertEqual(candidate.capture_timestamp, 3.5)
        self.assertEqual(legacy.event, "USER_TURN_END")
        self.assertEqual(adapter.emitted_events, [])

    def test_plural_candidate_mapping_uses_official_wrapper_shapes(self):
        """Catches the adapter accepting only a bespoke single-state mapping."""
        adapter = X2TurnAdapter(backend=object())
        first_frame = type("TurnFrame", (), {"label": "speaking", "score": 0.8})()
        second_frame = type("TurnFrame", (), {"label": "turn-end", "confidence": 0.7})()
        official_result = RuntimeResult("ASR is not authoritative here", [first_frame, second_frame])

        object_candidates = adapter.map_candidates(official_result)
        tuple_candidates = adapter.map_candidates(("ignored transcript", ["idle", "noidle"]))
        count_candidates = adapter.map_candidates({"turn_labels": {"speaking": 2, "turn_end": 1}})

        self.assertEqual([candidate.label for candidate in object_candidates], ["speaking", "turn_end"])
        self.assertEqual([candidate.label for candidate in tuple_candidates], ["idle", "noidle"])
        self.assertEqual(len(count_candidates), 1)
        self.assertEqual(count_candidates[0].label, "speaking")
        self.assertTrue(count_candidates[0].metadata["aggregated"])
        with self.assertRaisesRegex(ValueError, "map_candidates"):
            adapter.map_candidate(official_result)

        single = adapter.map_candidate({"label": "idle", "timestamp": 2.5, "sequence": 7})
        self.assertEqual(single.capture_timestamp, 2.5)
        self.assertEqual(single.sequence, 7)


class X2TurnBenchmarkScriptTests(unittest.TestCase):
    def test_help_runs_from_an_external_working_directory_without_loading_a_model(self):
        """Catches benchmark imports that only work from the repository root."""
        repository_root = Path(__file__).resolve().parents[1]
        script = repository_root / "scripts" / "benchmark_x2_turn_streaming.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=Path("/tmp"),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertIn("--x2-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
