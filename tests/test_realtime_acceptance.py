import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.metrics import (
    AcceptanceProvenance,
    EvidenceSource,
    evaluate_acceptance,
)
from benchmarks.real_hardware_runner import AcceptanceEvidence
from benchmarks.timeline import BenchmarkTimeline
from benchmarks.metrics import calculate_metrics
from scripts import run_realtime_acceptance


_PASSING_METRICS = {
    "duck_latency_ms": 90.0,
    "interrupt_latency_ms": 200.0,
    "backchannel_restore_latency_ms": 250.0,
    "first_token_latency_ms": 700.0,
    "first_audio_latency_ms": 1400.0,
    "first_translated_audio_latency_ms": 1900.0,
    "stale_output_count": 0.0,
    "resume_phrase_error_count": 1.0,
}


class RealtimeAcceptanceTests(unittest.TestCase):
    def test_acceptance_rejects_stale_audio_and_slow_interrupt(self):
        """Catches green reports that ignore hard latency or stale-output failures."""
        result = evaluate_acceptance(
            {
                **_PASSING_METRICS,
                "interrupt_latency_ms": 251.0,
                "stale_output_count": 1.0,
            },
            label="simulated",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.label, "simulated")
        self.assertEqual(set(result.failures), {"interrupt_latency_ms", "stale_output_count"})

    def test_high_tier_evidence_requires_matching_runner_provenance(self):
        """Catches imported metrics or the wrong driver being promoted to a real tier."""
        imported_model = evaluate_acceptance(
            _PASSING_METRICS,
            label="model-integration",
        )
        recorded = AcceptanceProvenance.runner_capture(
            EvidenceSource.RECORDED_AUDIO_REALTIME,
            run_id="recorded-run-1",
        )
        wrong_driver = evaluate_acceptance(
            _PASSING_METRICS,
            label="hardware-e2e",
            provenance=recorded,
        )

        self.assertFalse(imported_model.passed)
        self.assertFalse(wrong_driver.passed)
        self.assertIn("model_integration_requires_recorded_audio_runner", imported_model.failures)
        self.assertIn("hardware_e2e_requires_browser_headset_runner", wrong_driver.failures)

    def test_constructed_matching_provenance_cannot_mint_high_tier_evidence(self):
        """Catches public provenance fields acting as a high-tier capability."""
        model = evaluate_acceptance(
            _PASSING_METRICS,
            label="model-integration",
            provenance=AcceptanceProvenance(
                source=EvidenceSource.RECORDED_AUDIO_REALTIME,
                run_id="forged-recorded-run",
                producer=AcceptanceProvenance.RUNNER_PRODUCER,
            ),
        )
        hardware = evaluate_acceptance(
            _PASSING_METRICS,
            label="hardware-e2e",
            provenance=AcceptanceProvenance(
                source=EvidenceSource.BROWSER_HEADSET,
                run_id="forged-hardware-run",
                producer=AcceptanceProvenance.RUNNER_PRODUCER,
            ),
        )

        self.assertFalse(model.passed)
        self.assertFalse(hardware.passed)

    def test_first_audio_latency_starts_at_confirmed_turn_end(self):
        """Catches TTS-only latency that omits policy and LLM time before the first token."""
        timeline = BenchmarkTimeline()
        timeline.record("turn_end", 10.0)
        timeline.record("first_llm_token", 10.6)
        timeline.record("first_audio_chunk", 11.4)

        metrics = calculate_metrics(timeline)

        self.assertEqual(metrics["first_token_latency_ms"], 600.0)
        self.assertEqual(metrics["first_audio_latency_ms"], 1400.0)

    def test_missing_hard_gate_metrics_are_failures(self):
        """Catches partial reports that pass by omitting measurements."""
        result = evaluate_acceptance(
            {"stale_output_count": 0.0},
            label="simulated",
        )

        self.assertFalse(result.passed)
        self.assertIn("first_audio_latency_ms", result.failures)
        self.assertIn("resume_phrase_error_count", result.failures)

    def test_atomic_report_failure_preserves_previous_report(self):
        """Catches interrupted report replacement that truncates valid evidence."""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "acceptance.json"
            report.write_text("previous\n", encoding="utf-8")

            with patch.object(
                run_realtime_acceptance.os,
                "replace",
                side_effect=OSError("replacement interrupted"),
            ):
                with self.assertRaisesRegex(OSError, "replacement interrupted"):
                    run_realtime_acceptance.write_report_atomic(report, "replacement\n")

            self.assertEqual(report.read_text(encoding="utf-8"), "previous\n")
            self.assertEqual([path.name for path in Path(directory).iterdir()], [report.name])


class RealtimeAcceptanceCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_imported_json_is_limited_to_unit_and_simulated_labels(self):
        """Catches imported JSON self-attesting as model or hardware evidence."""
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metrics": _PASSING_METRICS,
                        "provenance": {
                            "source": EvidenceSource.BROWSER_HEADSET.value,
                            "producer": AcceptanceProvenance.RUNNER_PRODUCER,
                            "run_id": "forged-import",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = run_realtime_acceptance.parse_args(
                ["--label", "model-integration", "--metrics-json", str(metrics_path)]
            )

            with self.assertRaisesRegex(ValueError, "only unit or simulated"):
                await run_realtime_acceptance.run_acceptance(args)

    async def test_simulated_label_evaluates_imported_metrics_without_provenance(self):
        """Catches low-tier imported evaluation accidentally receiving runner provenance."""
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(json.dumps(_PASSING_METRICS), encoding="utf-8")
            args = run_realtime_acceptance.parse_args(
                ["--label", "simulated", "--metrics-json", str(metrics_path)]
            )

            payload = await run_realtime_acceptance.run_acceptance(args)

        self.assertTrue(payload["passed"])
        self.assertIsNone(payload["provenance"])

    async def test_plain_driver_return_value_cannot_mint_model_integration_evidence(self):
        """Catches injected drivers promoting a directly constructed evidence object."""
        calls = []
        timeline = BenchmarkTimeline()
        driver = _StaticAcceptanceDriver(
            AcceptanceEvidence(
                metrics=dict(_PASSING_METRICS),
                environment={"runtime": "task13"},
                provenance=AcceptanceProvenance.runner_capture(
                    EvidenceSource.RECORDED_AUDIO_REALTIME,
                    run_id="recorded-test",
                ),
                timeline=timeline,
            ),
            calls,
        )
        args = run_realtime_acceptance.parse_args(
            [
                "--label",
                "model-integration",
                "--profile",
                "local_gpu",
                "--audio",
                "/captures/turn.wav",
            ]
        )

        payload = await run_realtime_acceptance.run_acceptance(
            args,
            recorded_audio_driver=driver,
        )

        self.assertFalse(payload["passed"])
        self.assertIn(
            "model_integration_requires_recorded_audio_runner",
            payload["failures"],
        )
        self.assertEqual(calls, [("recorded", Path("/captures/turn.wav"), "local_gpu")])
        self.assertEqual(
            payload["provenance"]["source"],
            EvidenceSource.RECORDED_AUDIO_REALTIME.value,
        )

    async def test_hardware_e2e_requires_browser_headset_driver(self):
        """Catches command-line flags substituting for a physical capture driver."""
        args = run_realtime_acceptance.parse_args(["--label", "hardware-e2e"])

        with self.assertRaisesRegex(ValueError, "browser/headset capture driver"):
            await run_realtime_acceptance.run_acceptance(args)


class _StaticAcceptanceDriver:
    def __init__(self, evidence, calls):
        self.evidence = evidence
        self.calls = calls

    async def run(self, audio_path=None, *, profile):
        if audio_path is None:
            self.calls.append(("browser", profile))
        else:
            self.calls.append(("recorded", Path(audio_path), profile))
        return self.evidence


if __name__ == "__main__":
    unittest.main()
