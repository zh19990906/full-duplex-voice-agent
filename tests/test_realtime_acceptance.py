import unittest

from benchmarks.metrics import evaluate_acceptance


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

    def test_hardware_e2e_requires_explicit_real_run_and_required_measurements(self):
        """Catches synthetic or partial evidence being mislabeled as hardware E2E."""
        not_real = evaluate_acceptance(
            _PASSING_METRICS,
            label="hardware-e2e",
            explicit_real_run=False,
            required_measurements_present=True,
        )
        missing_measurements = evaluate_acceptance(
            _PASSING_METRICS,
            label="hardware-e2e",
            explicit_real_run=True,
            required_measurements_present=False,
        )

        self.assertFalse(not_real.passed)
        self.assertFalse(missing_measurements.passed)
        self.assertIn("hardware_e2e_requires_explicit_real_run", not_real.failures)
        self.assertIn("hardware_e2e_missing_required_measurements", missing_measurements.failures)


if __name__ == "__main__":
    unittest.main()
