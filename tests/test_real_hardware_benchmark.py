import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.real_hardware_runner import (
    HardwareValidationError,
    HardwareValidationResult,
    RealHardwareBenchmarkRunner,
    validate_hardware,
)
from benchmarks.report import HardwareBenchmarkReport


def benchmark_config(model_root: str = "./models"):
    return {
        "audio": {
            "input_provider": "sounddevice",
            "output_provider": "sounddevice",
        },
        "models": {
            "model_root": model_root,
            "asr": {"provider": "whisper", "local_path": "./models/asr", "device": "cuda"},
            "llm": {"provider": "llama_cpp", "local_path": "./models/llm", "device": "cuda"},
            "tts": {"provider": "cosyvoice", "local_path": "./models/tts", "device": "cpu"},
        },
        "runtime": {"environment": "hardware_benchmark"},
    }


class FakeHardwareProbe:
    def __init__(self, microphone=True, speaker=True, gpu="Test GPU"):
        self.microphone = microphone
        self.speaker = speaker
        self.gpu = gpu

    def check_microphone(self, audio):
        return self.microphone

    def check_speaker(self, audio):
        return self.speaker

    def check_model_path(self, path):
        return True

    def gpu_info(self, config):
        return {"available": self.gpu is not None, "name": self.gpu}


class StaticLoader:
    def __init__(self, config):
        self.config = config

    def load(self, profile):
        return self.config


class FakeRuntime:
    def __init__(self):
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        self.shutdown_called = True
        self.initialized = False


class RealHardwareBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_device_detection_is_explicit(self):
        result = validate_hardware(
            benchmark_config(),
            probe=FakeHardwareProbe(microphone=False),
        )

        self.assertFalse(result.valid)
        self.assertIn("microphone device unavailable", result.errors)
        with self.assertRaises(HardwareValidationError):
            result.require_valid()

    def test_missing_model_detection_is_explicit(self):
        class MissingModelProbe(FakeHardwareProbe):
            def check_model_path(self, path):
                return path.name != "llm"

        result = validate_hardware(benchmark_config(), probe=MissingModelProbe())

        self.assertFalse(result.valid)
        self.assertIn("missing model path: llm", result.errors)

    def test_cuda_requirement_is_validated(self):
        result = validate_hardware(
            benchmark_config(),
            probe=FakeHardwareProbe(gpu=None),
        )

        self.assertFalse(result.valid)
        self.assertIn("GPU unavailable but CUDA model is configured", result.errors)

    async def test_runner_starts_runtime_and_collects_real_timeline_events(self):
        runtime = FakeRuntime()

        def runtime_factory(profile, config):
            return runtime

        async def session_hook(active_runtime, timeline):
            self.assertIs(active_runtime, runtime)
            timeline.record("audio_received", 10.0)
            timeline.record("first_asr_partial", 10.1)
            timeline.record("turn_end", 10.2)
            timeline.record("first_llm_token", 10.4)
            timeline.record("first_audio_chunk", 10.7)

        runner = RealHardwareBenchmarkRunner(
            profile="hardware_benchmark",
            config_loader=StaticLoader(benchmark_config()),
            probe=FakeHardwareProbe(),
            runtime_factory=runtime_factory,
        )
        report = await runner.run(session_hook)

        self.assertTrue(runtime.shutdown_called)
        self.assertIn("first_transcript_latency_ms", report.metrics)
        self.assertIn("first_token_latency_ms", report.metrics)
        self.assertIn("first_audio_latency_ms", report.metrics)
        self.assertEqual(report.environment["profile"], "hardware_benchmark")

    def test_report_supports_json_and_human_output(self):
        report = HardwareBenchmarkReport(
            metrics={"first_token_latency_ms": 200.0},
            environment={"profile": "hardware_benchmark", "gpu": "Test GPU"},
            system={"memory_mb": 42.0},
        )

        payload = json.loads(report.to_json())

        self.assertEqual(payload["environment"]["gpu"], "Test GPU")
        self.assertIn("Real Hardware Benchmark Report", report.to_text())
        self.assertIn("LLM TTFT: 200.00 ms", report.to_text())


if __name__ == "__main__":
    unittest.main()
