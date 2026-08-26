import asyncio
import unittest

from src.asr.stream import TranscriptChunk
from src.controller.actions import ActionType, ControllerAction
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import AudioFrameHeader
from src.runtime_app.container import (
    ServerRealtimeSessionRuntime,
    SlowConsumerError,
    build_production_runtime_factory,
)


MODEL_CONFIG = {
    "models": {
        "asr": {
            "provider": "local",
            "model_path": "/models/faster-whisper",
            "device": "cpu",
            "options": {"cadence_ms": 200},
        },
        "turn": {
            "provider": "local",
            "model_path": "/models/x2-turn",
            "device": "cpu",
            "options": {"cadence_ms": 100},
        },
        "policy": {
            "provider": "local",
            "model_path": "/models/policy-qwen",
            "device": "cpu",
        },
        "llm": {
            "provider": "transformers",
            "model_path": "/models/chat-qwen",
            "device": "cpu",
        },
        "tts": {
            "provider": "cosyvoice_worker",
            "model_path": "/models/cosyvoice",
            "device": "cpu",
        },
        "translation": {
            "provider": "transformers",
            "model_path": "/models/translate-qwen",
            "device": "cpu",
        },
    }
}


class _SharedAsrRuntime:
    def transcribe(self, _audio, **_options):
        return ""


class _SharedTurnRuntime:
    def infer(self, _audio):
        return ()


class _SharedPolicyRuntime:
    def generate(self, _prompt, **_options):
        return (
            '{"action":"UNCERTAIN","confidence":0.0,"rationale":"no-op"}'
        )


class _SharedLlmRuntime:
    async def stream_tokens(self, _prompt, **_options):
        if False:
            yield None

    async def generate(self, _prompt, **_options):
        return ""


class _SharedTranslationRuntime:
    def __init__(self, text="Hello"):
        self.text = text
        self.prompts = []

    async def generate(self, prompt, **_options):
        self.prompts.append(prompt)
        return self.text


class _SharedTtsRuntime:
    def __init__(self):
        self.requests = []

    async def stream_audio(self, text, **options):
        self.requests.append((text, dict(options)))
        yield b"\x00\x00" * 2


class _FakeStreamingAsr:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.cancelled = False

    async def push_pcm(self, _frame):
        if self.outputs:
            return self.outputs.pop(0)
        return None

    async def finalize_turn(self):
        return TranscriptChunk("final-empty", "", 2.0, True, revision_id=2)

    async def cancel(self):
        self.cancelled = True

    def reset(self):
        self.cancelled = False


class _FakeTranslationRuntime:
    def __init__(self, translated_text="Hello"):
        self.translated_text = translated_text
        self.prompts = []

    async def generate(self, prompt, **_options):
        self.prompts.append(prompt)
        return self.translated_text


class _FakeTtsRuntime:
    def __init__(self):
        self.requests = []

    async def stream_audio(self, text, **options):
        self.requests.append((text, dict(options)))
        yield b"\x00\x00" * 4


class RuntimeAppContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_production_factory_creates_real_shape_session_from_injected_loaded_runtimes(self):
        shared = {
            "asr": _SharedAsrRuntime(),
            "turn": _SharedTurnRuntime(),
            "policy": _SharedPolicyRuntime(),
            "llm": _SharedLlmRuntime(),
            "tts": _SharedTtsRuntime(),
            "translation": _SharedTranslationRuntime(),
        }
        factory = build_production_runtime_factory(
            model_config=MODEL_CONFIG,
            runtime_loaders={name: (lambda _config, runtime=runtime: runtime) for name, runtime in shared.items()},
        )

        session = factory("session-smoke")

        self.assertIsInstance(session, ServerRealtimeSessionRuntime)
        self.assertIsNotNone(session.asr)
        self.assertIsNotNone(session.turn)
        self.assertIsNotNone(session.policy_engine)
        self.assertIsNotNone(session.interpretation_translator)
        await session.close()
        await factory.close()

    async def test_production_factory_shares_only_loaded_runtimes_and_isolates_stateful_wrappers(self):
        shared = {
            "asr": _SharedAsrRuntime(),
            "turn": _SharedTurnRuntime(),
            "policy": _SharedPolicyRuntime(),
            "llm": _SharedLlmRuntime(),
            "tts": _SharedTtsRuntime(),
            "translation": _SharedTranslationRuntime(),
        }
        factory = build_production_runtime_factory(
            model_config=MODEL_CONFIG,
            runtime_loaders={name: (lambda _config, runtime=runtime: runtime) for name, runtime in shared.items()},
        )

        first = factory("session-a")
        second = factory("session-b")

        self.assertIsNot(first.asr, second.asr)
        self.assertIs(first.asr.runtime.runtime, second.asr.runtime.runtime)
        self.assertIsNot(first.turn, second.turn)
        self.assertIs(first.turn.runtime.runtime, second.turn.runtime.runtime)
        self.assertIsNot(first.llm, second.llm)
        self.assertIs(first.llm.provider.runtime.runtime, second.llm.provider.runtime.runtime)
        self.assertIsNot(first.tts, second.tts)
        self.assertIs(first.tts.provider.runtime.runtime, second.tts.provider.runtime.runtime)
        self.assertIsNot(first.policy_engine, second.policy_engine)
        self.assertIs(
            first.policy_engine.provider.runtime.runtime,
            second.policy_engine.provider.runtime.runtime,
        )
        self.assertIsNot(first.interpretation_translator, second.interpretation_translator)
        self.assertIs(
            first.interpretation_translator.runtime.runtime,
            second.interpretation_translator.runtime.runtime,
        )

        await first.asr.cancel()
        await first.turn.cancel()
        await first.llm.cancel()
        await first.tts.interrupt()
        first.policy_engine.provider.runtime.cancelled = True

        self.assertTrue(first.asr._cancelled)
        self.assertFalse(second.asr._cancelled)
        self.assertTrue(first.turn._cancelled)
        self.assertFalse(second.turn._cancelled)
        self.assertTrue(first.llm._cancelled)
        self.assertFalse(second.llm._cancelled)
        self.assertTrue(first.tts._interrupted)
        self.assertFalse(second.tts._interrupted)

        await first.close()
        await second.close()
        await factory.close()

    async def test_interpretation_mode_routes_committed_asr_chunks_through_translation_tts_and_events(self):
        translator_runtime = _FakeTranslationRuntime("Hello")
        tts_runtime = _FakeTtsRuntime()
        runtime = ServerRealtimeSessionRuntime(
            "session-interpret",
            llm=_SharedLlmRuntime(),
            tts=type("TTS", (), {"stream_audio": tts_runtime.stream_audio, "interrupt": lambda self: None, "reset": lambda self: None})(),
            asr=_FakeStreamingAsr(
                [TranscriptChunk("chunk-1", "你好", 1.0, False, revision_id=1)]
            ),
            interpretation_translator=type(
                "Translator",
                (),
                {
                    "__init__": lambda self, runtime: setattr(self, "runtime", runtime),
                    "translate_stream": lambda self, prompt: self.runtime.generate(prompt),
                },
            )(translator_runtime),
        )
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": "Chinese",
                        "target_language": "English",
                    },
                ),
            )
        )

        await runtime.accept_audio_frame(
            RealtimeAudioFrame(
                AudioFrameHeader(sequence=0, capture_timestamp=0.0),
                b"\x00\x00" * 320,
            )
        )
        await runtime.close()

        events = []
        async for item in runtime.events():
            events.append(item)

        self.assertEqual(events[0]["event"], "AUDIO_FRAME_ACCEPTED")
        self.assertEqual(events[1]["event"], "transcript")
        self.assertEqual(events[2]["event"], "translation")
        self.assertEqual(events[2]["payload"]["translated_text"], "Hello")
        self.assertEqual(events[3]["event"], "audio_chunk")
        self.assertEqual(tts_runtime.requests[0][0], "Hello")
        checkpoint = runtime.checkpoints.get(runtime.current_response_id)
        self.assertEqual(checkpoint.segments[0].text, "Hello")
        self.assertEqual(checkpoint.segments[0].audio, b"\x00\x00" * 4)
        self.assertEqual(len(translator_runtime.prompts), 1)

    async def test_outbound_event_queue_full_closes_runtime_with_explicit_slow_consumer_error(self):
        runtime = ServerRealtimeSessionRuntime(
            "session-backpressure",
            llm=_SharedLlmRuntime(),
            tts=type("TTS", (), {"stream_audio": _FakeTtsRuntime().stream_audio, "interrupt": lambda self: None, "reset": lambda self: None})(),
            outbound_event_queue_capacity=1,
        )

        await runtime._publish({"event": "first", "payload": {"value": 1}})
        with self.assertRaises(SlowConsumerError):
            await runtime._publish({"event": "second", "payload": {"value": 2}})

        self.assertTrue(runtime.closed)
        events = []
        with self.assertRaises(SlowConsumerError):
            async for item in runtime.events():
                events.append(item)
        self.assertEqual(events, [{"event": "first", "payload": {"value": 1}}])
