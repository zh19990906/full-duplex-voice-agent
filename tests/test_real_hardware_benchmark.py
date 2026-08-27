import asyncio
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from benchmarks.real_hardware_runner import (
    BrowserHeadsetAcceptanceDriver,
    HardwareValidationError,
    HardwareValidationResult,
    RecordedAudioRealtimeDriver,
    RealHardwareBenchmarkRunner,
    validate_hardware,
)
from benchmarks.metrics import EvidenceSource, evaluate_acceptance
from src.realtime.audio_ingress import RealtimeAudioFrame
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


class RecordingRealtimeRuntime:
    def __init__(self):
        self.frames = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def accept_audio_frame(self, frame):
        self.frames.append(frame)

    async def flush(self):
        return None

    async def close(self):
        self.closed = True

    async def events(self):
        yield {"event": "turn_end", "server_timestamp": 1.0, "payload": {}}
        yield {"event": "token", "server_timestamp": 1.2, "payload": {}}
        yield {
            "event": "audio_chunk",
            "server_timestamp": 1.5,
            "response_id": "response-1",
            "generation_epoch": 1,
            "segment_id": 0,
            "playback_attempt_id": 1,
            "payload": {"audio_data": b"\x01\x00"},
        }


class RecordingRuntimeFactory:
    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = False

    def __call__(self, _session_id):
        return self.runtime

    async def close(self):
        self.closed = True


class MutableClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class TurnBoundaryController:
    def handle_candidate(self, _state, _candidate):
        return ()


class ConfirmedTurnRuntime(RecordingRealtimeRuntime):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.controller = TurnBoundaryController()

    async def accept_audio_frame(self, frame):
        self.frames.append(frame)
        self.clock.value = 1.0
        self.controller.handle_candidate(
            None,
            SimpleNamespace(event="USER_TURN_END_CANDIDATE"),
        )
        self.clock.value = 1.8

    async def events(self):
        yield {"event": "token", "server_timestamp": 1.2, "payload": {}}
        yield {
            "event": "audio_chunk",
            "server_timestamp": 2.0,
            "response_id": "response-1",
            "generation_epoch": 1,
            "segment_id": 0,
            "playback_attempt_id": 1,
            "payload": {"audio_data": b"\x01\x00"},
        }


class NoTurnEndRuntime(RecordingRealtimeRuntime):
    async def events(self):
        yield {"event": "token", "server_timestamp": 1.2, "payload": {}}


class SentinelAudioRuntime(RecordingRealtimeRuntime):
    async def events(self):
        yield {"event": "turn_end", "server_timestamp": 1.0, "payload": {}}
        yield {
            "event": "audio_chunk",
            "server_timestamp": 1.1,
            "response_id": "response-1",
            "generation_epoch": 1,
            "segment_id": 0,
            "playback_attempt_id": 1,
            "payload": {"audio_data": b""},
        }
        yield {
            "event": "audio_chunk",
            "server_timestamp": 1.2,
            "payload": {"audio_data": b"\x01\x00"},
        }
        yield {
            "event": "audio_chunk",
            "server_timestamp": 1.5,
            "response_id": "response-1",
            "generation_epoch": 1,
            "segment_id": 0,
            "playback_attempt_id": 1,
            "payload": {"audio_data": b"\x01\x00"},
        }


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

    async def test_recorded_audio_driver_feeds_task13_runtime_in_realtime_frames(self):
        """Catches --audio being retained as metadata instead of entering the realtime runtime."""
        runtime = RecordingRealtimeRuntime()
        factory = RecordingRuntimeFactory(runtime)
        sleep_delays = []
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "turn.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 640)
            driver = RecordedAudioRealtimeDriver(
                runtime_factory=lambda _profile: factory,
                sleep=lambda delay: sleep_delays.append(delay),
                drain_timeout=0.01,
                trailing_silence_frames=2,
            )

            evidence = await driver.run(audio_path, profile="local_gpu")

        self.assertTrue(runtime.started)
        self.assertTrue(runtime.closed)
        self.assertTrue(factory.closed)
        self.assertEqual([frame.header.sequence for frame in runtime.frames], [0, 1, 2, 3])
        self.assertEqual([frame.pcm for frame in runtime.frames[-2:]], [b"\x00" * 640] * 2)
        self.assertTrue(all(isinstance(frame, RealtimeAudioFrame) for frame in runtime.frames))
        self.assertEqual(sleep_delays, [0.02, 0.02, 0.02])
        self.assertEqual(evidence.provenance.source, EvidenceSource.RECORDED_AUDIO_REALTIME)
        self.assertEqual(evidence.metrics["first_audio_latency_ms"], 500.0)

    async def test_recorded_audio_driver_observes_confirmed_task13_turn_end(self):
        """Catches EOF timing replacing the fused Task13 turn-end boundary."""
        clock = MutableClock()
        runtime = ConfirmedTurnRuntime(clock)
        factory = RecordingRuntimeFactory(runtime)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "turn.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 320)
            driver = RecordedAudioRealtimeDriver(
                runtime_factory=lambda _profile: factory,
                sleep=lambda _delay: None,
                drain_timeout=0.01,
                clock=clock,
                trailing_silence_frames=1,
            )

            evidence = await driver.run(audio_path, profile="local_gpu")

        self.assertEqual(evidence.timeline.first("turn_end").timestamp, 1.0)
        self.assertEqual(evidence.metrics["first_audio_latency_ms"], 1000.0)

    async def test_recorded_audio_rejects_capture_without_fused_turn_end(self):
        runtime = NoTurnEndRuntime()
        factory = RecordingRuntimeFactory(runtime)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "turn.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x01\x00" * 320)
            driver = RecordedAudioRealtimeDriver(
                runtime_factory=lambda _profile: factory,
                sleep=lambda _delay: None,
                drain_timeout=0.01,
                trailing_silence_frames=2,
            )

            with self.assertRaisesRegex(RuntimeError, "fused turn_end"):
                await driver.run(audio_path, profile="local_gpu")

        self.assertEqual(len(runtime.frames), 3)
        self.assertTrue(runtime.closed)
        self.assertTrue(factory.closed)

    async def test_first_playable_audio_ignores_empty_and_identityless_chunks(self):
        runtime = SentinelAudioRuntime()
        factory = RecordingRuntimeFactory(runtime)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "turn.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x01\x00" * 320)
            driver = RecordedAudioRealtimeDriver(
                runtime_factory=lambda _profile: factory,
                sleep=lambda _delay: None,
                drain_timeout=0.01,
                trailing_silence_frames=1,
            )

            evidence = await driver.run(audio_path, profile="local_gpu")

        self.assertEqual(evidence.timeline.count("first_audio_chunk"), 1)
        self.assertEqual(evidence.metrics["first_audio_latency_ms"], 500.0)

    async def test_browser_headset_driver_executes_driver_and_stamps_runner_provenance(self):
        """Catches hardware E2E reports produced without running the browser/headset driver."""
        calls = []

        async def run_browser(profile):
            calls.append(profile)
            return {
                "timeline": [
                    {"name": "turn_end", "timestamp": 2.0},
                    {
                        "name": "first_audio_chunk",
                        "timestamp": 2.4,
                        "response_id": "response-1",
                        "generation_epoch": 1,
                        "segment_id": 0,
                        "playback_attempt_id": 1,
                        "pcm": [1, 0],
                    },
                ],
                "metrics": {"resume_phrase_error_count": 0.0},
                "environment": {"headset": "usb"},
            }

        evidence = await BrowserHeadsetAcceptanceDriver(run_browser).run(profile="local_gpu")

        self.assertEqual(calls, ["local_gpu"])
        self.assertEqual(evidence.provenance.source, EvidenceSource.BROWSER_HEADSET)
        self.assertEqual(evidence.metrics["first_audio_latency_ms"], 400.0)
        self.assertEqual(evidence.environment["headset"], "usb")
        result = evaluate_acceptance(
            {
                "duck_latency_ms": 90.0,
                "interrupt_latency_ms": 200.0,
                "backchannel_restore_latency_ms": 250.0,
                "first_token_latency_ms": 700.0,
                "first_audio_latency_ms": 400.0,
                "first_translated_audio_latency_ms": 1900.0,
                "stale_output_count": 0.0,
                "resume_phrase_error_count": 1.0,
            },
            label="hardware-e2e",
            provenance=evidence.provenance,
            _evidence_capability=evidence._capability,
        )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
