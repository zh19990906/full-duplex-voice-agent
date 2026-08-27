import json
import unittest

from benchmarks.metrics import calculate_metrics
from benchmarks.report import BenchmarkReport
from benchmarks.runner import run_benchmark
from benchmarks.scenarios import load_scenarios
from benchmarks.timeline import BenchmarkTimeline


class BenchmarkValidationTests(unittest.TestCase):
    def test_timeline_preserves_order_and_serializes(self):
        timeline = BenchmarkTimeline()
        timeline.record("user_audio_received", 1.0)
        timeline.record("asr_partial_received", 1.25)
        timeline.record("turn_end", 2.0)
        self.assertEqual(timeline.names(), ["user_audio_received", "asr_partial_received", "turn_end"])
        self.assertEqual(timeline.to_dict()["events"][1]["timestamp"], 1.25)

    def test_metrics_are_calculated_in_milliseconds(self):
        timeline = BenchmarkTimeline()
        for name, timestamp in (
            ("audio_received", 1.0),
            ("first_asr_partial", 1.12),
            ("turn_end", 2.0),
            ("first_llm_token", 2.15),
            ("first_audio_chunk", 2.40),
            ("user_interrupt", 3.0),
            ("tts_stopped", 3.08),
            ("cancel_requested", 3.1),
            ("generation_cancelled", 3.15),
        ):
            timeline.record(name, timestamp)
        metrics = calculate_metrics(timeline)
        self.assertEqual(metrics["first_transcript_latency_ms"], 120.0)
        self.assertEqual(metrics["first_token_latency_ms"], 150.0)
        self.assertEqual(metrics["first_audio_latency_ms"], 400.0)
        self.assertEqual(metrics["interrupt_latency_ms"], 80.0)
        self.assertEqual(metrics["cancellation_latency_ms"], 50.0)

    def test_interrupt_benchmark_runner(self):
        timeline = BenchmarkTimeline()
        timeline.record("user_interrupt", 10.0)
        timeline.record("tts_stopped", 10.05)
        timeline.record("cancel_requested", 10.01)
        timeline.record("generation_cancelled", 10.07)
        result = run_benchmark("interrupt", timeline)
        self.assertEqual(result.scenario, "interrupt")
        self.assertEqual(result.metrics["interrupt_latency_ms"], 50.0)
        self.assertEqual(result.metrics["cancellation_latency_ms"], 60.0)

    def test_report_supports_json_and_human_output(self):
        report = BenchmarkReport("interrupt", {"interrupt_latency_ms": 80.0})
        payload = json.loads(report.to_json())
        self.assertEqual(payload["scenario"], "interrupt")
        self.assertIn("Interrupt latency: 80.00 ms", report.to_text())

    def test_existing_integration_scenarios_are_reused(self):
        scenarios = load_scenarios()
        self.assertEqual(len(scenarios), 4)
        self.assertEqual([scenario.scenario_id for scenario in scenarios], [
            "case01",
            "case02",
            "case03",
            "case04",
        ])


if __name__ == "__main__":
    unittest.main()
