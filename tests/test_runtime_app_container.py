import asyncio
import unittest
from unittest.mock import patch

from src.asr.stream import TranscriptChunk
from src.controller.actions import ActionType, ControllerAction
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import AudioFrameHeader
from src.runtime_app.container import (
    _SharedRuntimeBoundary,
    _SessionRuntimeProxy,
    _default_runtime_loaders,
    _effective_model_config,
    _load_x2_runtime_from_profile,
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


class _BlockingRuntime:
    def __init__(self):
        self.active = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_calls = []
        self.interrupt_calls = []
        self.reset_calls = 0

    async def stream_tokens(self, _prompt, **_options):
        self.active.set()
        try:
            await self.release.wait()
        finally:
            pass
        yield {"text": "done", "is_final": True}

    async def stream_audio(self, _text, **_options):
        self.active.set()
        await self.release.wait()
        yield b"\x00\x00"

    async def cancel(self, *args):
        self.cancel_calls.append(args)
        self.release.set()

    async def interrupt(self, *args):
        self.interrupt_calls.append(args)
        self.release.set()

    def reset(self):
        self.reset_calls += 1


class _LoaderCapture:
    def __init__(self):
        self.calls = []

    def __call__(self, **options):
        self.calls.append(dict(options))
        return {"backend_options": dict(options)}


class RuntimeAppContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_llm_cancel_reaches_shared_runtime_while_stream_is_blocked(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        proxy = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in proxy.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()

        await proxy.cancel()
        result = await task

        self.assertEqual(runtime.cancel_calls, [()])
        self.assertEqual([chunk["text"] for chunk in result], ["done"])

    async def test_owner_tts_cancel_preserves_request_id_and_reaches_runtime_promptly(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        proxy = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in proxy.stream_audio("hello", request_id="req-7")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()

        await proxy.cancel("req-7")
        await task

        self.assertEqual(runtime.cancel_calls, [("req-7",)])

    async def test_waiting_session_cannot_cancel_another_session_owner(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        owner = _SessionRuntimeProxy(boundary)
        waiting = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in owner.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()

        await waiting.cancel("foreign-request")
        self.assertEqual(runtime.cancel_calls, [])

        await owner.cancel()
        await task
        self.assertEqual(runtime.cancel_calls, [()])

    async def test_reset_forwards_when_boundary_becomes_idle_without_deadlock(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        owner = _SessionRuntimeProxy(boundary)
        waiter = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in owner.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()
        await owner.cancel()
        await task

        reset_result = waiter.reset()
        if asyncio.isfuture(reset_result) or asyncio.iscoroutine(reset_result):
            await reset_result
        await asyncio.sleep(0)

        self.assertEqual(runtime.reset_calls, 1)

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

    def test_default_turn_loader_honors_resolved_profile_values(self):
        loader = _LoaderCapture()
        x2_module = type("X2Module", (), {"create": loader})()
        profile = {
            "model_path": "/resolved/turn-model",
            "local_path": "/ignored/local-model",
            "device": "cpu",
            "options": {"cadence_ms": 120, "context_seconds": 1.5, "beam_size": 4},
        }

        with patch("importlib.import_module", return_value=x2_module):
            loaded = _load_x2_runtime_from_profile(profile)

        self.assertEqual(
            loader.calls,
            [
                {
                    "model_path": "/resolved/turn-model",
                    "device": "cpu",
                    "cadence_ms": 120,
                    "context_seconds": 1.5,
                    "beam_size": 4,
                }
            ],
        )
        self.assertEqual(loaded["backend_options"]["beam_size"], 4)

    def test_default_turn_loader_is_wired_into_factory_runtime_loaders(self):
        effective = _effective_model_config(MODEL_CONFIG, {"turn_model": "/cli/turn-model"})
        capture = {}

        def turn_loader(profile):
            capture.update(profile)
            return _SharedTurnRuntime()

        factory = build_production_runtime_factory(
            model_config=MODEL_CONFIG,
            overrides={"turn_model": "/cli/turn-model"},
            runtime_loaders={
                "asr": lambda _profile: _SharedAsrRuntime(),
                "turn": turn_loader,
                "policy": lambda _profile: _SharedPolicyRuntime(),
                "llm": lambda _profile: _SharedLlmRuntime(),
                "tts": lambda _profile: _SharedTtsRuntime(),
                "translation": lambda _profile: _SharedTranslationRuntime(),
            },
        )

        session = factory("session-turn-loader")

        self.assertEqual(capture["model_path"], "/cli/turn-model")
        self.assertEqual(capture["local_path"], "/cli/turn-model")
        self.assertEqual(capture["device"], effective["turn"]["device"])
        self.assertIsNotNone(session.turn)

    def test_default_turn_profile_includes_bounded_runtime_options(self):
        effective = _effective_model_config({}, {})

        self.assertEqual(effective["turn"]["provider"], "local")
        self.assertEqual(effective["turn"]["model_path"], "./models/turn")
        self.assertEqual(effective["turn"]["local_path"], "./models/turn")
        self.assertEqual(effective["turn"]["device"], "cuda")
        self.assertEqual(effective["turn"]["options"]["cadence_ms"], 160)
        self.assertEqual(effective["turn"]["options"]["context_seconds"], 2.0)
