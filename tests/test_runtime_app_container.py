import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.asr.stream import TranscriptChunk
from src.adapters.turn.x2_turn_streaming import TurnCandidate
from src.controller.actions import ActionType, ControllerAction
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import AudioFrameHeader
from src.realtime.response_pipeline import ResponseStreamEnd
from src.realtime.speech_fusion import SpeechCandidateEvent, SpeechEventFusion
from src.runtime_app.container import (
    _build_model_manager,
    _SharedRuntimeBoundary,
    _SessionRuntimeProxy,
    _default_runtime_loaders,
    _effective_model_config,
    _load_x2_runtime_from_profile,
    _measure_cuda_allocated_bytes,
    ServerRealtimeSessionRuntime,
    SlowConsumerError,
    build_production_runtime_factory,
)
from src.model_runtime.manager import ModelMemoryBudgetError
from src.realtime.cancellation import CancellationToken
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.text_segmenter import TextSegment
from src.llm_runtime.stream import TokenChunk


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


class _RecordingModePolicy:
    def __init__(self, decision):
        self.decision = decision
        self.requests = []

    async def decide(self, request):
        self.requests.append(request)
        return self.decision


class _SharedLlmRuntime:
    async def stream_tokens(self, _prompt, **_options):
        if False:
            yield None

    async def generate(self, _prompt, **_options):
        return ""


class _ContextPromptLlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def stream_tokens(self, prompt, **_options):
        self.prompts.append(prompt)
        yield TokenChunk("captured", next(self.responses), 1.0, True)

    async def cancel(self):
        return None

    def reset(self):
        return None


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


class _ReloadableTtsRuntime:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.requests = []
        self.closed = False

    async def stream_audio(self, text, **options):
        self.requests.append((text, dict(options)))
        if self.fail:
            raise RuntimeError("tts worker crashed")
        yield b"\x00\x00" * 2

    async def close(self):
        self.closed = True


class _ClosableLoadedRuntime:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _CloseRecordingRuntime:
    def __init__(self, name, closed, *, error=None):
        self.name = name
        self.closed = closed
        self.error = error

    async def close(self):
        self.closed.append(self.name)
        if self.error is not None:
            raise self.error


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


class _BlockingStreamingAsr:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def push_pcm(self, _frame):
        self.started.set()
        await self.release.wait()
        return None

    async def finalize_turn(self):
        return TranscriptChunk("blocking-final", "", 1.0, True, revision_id=1)


class _BlockingTurn:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def push_pcm(self, _frame):
        self.started.set()
        await self.release.wait()
        return ()


class _SequencedTurn:
    def __init__(self, end_on_call=2):
        self.calls = 0
        self.end_on_call = end_on_call

    async def push_pcm(self, _frame):
        self.calls += 1
        if self.calls == self.end_on_call:
            return (TurnCandidate("turn_end"),)
        return ()


class _PromptRecordingLlm:
    def __init__(self):
        self.prompts = []
        self.called = asyncio.Event()

    async def stream_tokens(self, prompt, **_options):
        self.prompts.append(prompt)
        self.called.set()
        yield {"text": "ok", "is_final": True}

    def reset(self):
        return None

    async def interrupt(self):
        return None


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


class _IdleControlRaceRuntime:
    def __init__(self):
        self.control_started = asyncio.Event()
        self.allow_control_finish = asyncio.Event()
        self.operation_started = asyncio.Event()
        self.cancel_calls = []
        self.reset_calls = 0
        self.generate_calls = []

    async def cancel(self, *args):
        self.cancel_calls.append(args)
        self.control_started.set()
        await self.allow_control_finish.wait()

    async def reset(self):
        self.reset_calls += 1
        self.control_started.set()
        await self.allow_control_finish.wait()

    async def generate(self, prompt, **_options):
        self.generate_calls.append(prompt)
        self.operation_started.set()
        return f"generated:{prompt}"


class _ExplodingControlRuntime:
    def __init__(self):
        self.operation_started = asyncio.Event()
        self.operation_release = asyncio.Event()

    async def stream_tokens(self, _prompt, **_options):
        self.operation_started.set()
        await self.operation_release.wait()
        yield {"text": "done", "is_final": True}

    async def cancel(self, *_args):
        self.operation_release.set()
        raise RuntimeError("cancel boom")

    async def generate(self, prompt, **_options):
        return prompt


class _LoaderCapture:
    def __init__(self):
        self.calls = []

    def __call__(self, **options):
        self.calls.append(dict(options))
        return {"backend_options": dict(options)}


class RuntimeAppContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_paused_response_gates_future_tts_until_resume(self):
        tts = _SharedTtsRuntime()
        runtime = ServerRealtimeSessionRuntime(
            "session-pause-gate",
            llm=_SharedLlmRuntime(),
            tts=tts,
        )
        runtime.activate_response("response-paused")
        await runtime.apply_controller_actions(
            (ControllerAction(ActionType.PAUSE_RESPONSE),)
        )
        queue = asyncio.Queue()
        await queue.put(TextSegment(0, "稍后继续。", False, "response-paused", 0))
        await queue.put(ResponseStreamEnd("response-paused", 0, 0))
        consumer = asyncio.create_task(
            runtime._consume_segments(queue, 0, CancellationToken())
        )

        await asyncio.sleep(0.02)
        self.assertEqual(tts.requests, [])

        await runtime.apply_controller_actions(
            (ControllerAction(ActionType.RESUME_RESPONSE),)
        )
        await asyncio.wait_for(consumer, 0.2)
        self.assertEqual([request[0] for request in tts.requests], ["稍后继续。"])
        await runtime.close()

    async def test_activity_duck_fast_path_does_not_wait_for_asr_or_turn_models(self):
        asr = _BlockingStreamingAsr()
        turn = _BlockingTurn()
        runtime = ServerRealtimeSessionRuntime(
            "session-fast-activity",
            llm=_SharedLlmRuntime(),
            tts=_SharedTtsRuntime(),
            asr=asr,
            turn=turn,
        )
        runtime.activate_response("response-playing")
        runtime.session_state.response = runtime.session_state.response.PLAYING
        runtime.session_state.floor = runtime.session_state.floor.ASSISTANT
        frame = RealtimeAudioFrame(
            AudioFrameHeader(sequence=0, capture_timestamp=1.0),
            b"\xff\x7f" * 320,
        )

        accept_task = asyncio.create_task(runtime.accept_audio_frame(frame))
        try:
            await asyncio.wait_for(asr.started.wait(), 0.1)
            await asyncio.wait_for(turn.started.wait(), 0.1)
            await asyncio.wait_for(accept_task, 0.1)
            events = []
            while len(events) < 2:
                events.append(await asyncio.wait_for(anext(runtime.events()), 0.1))

            self.assertEqual(events[0]["event"], "AUDIO_FRAME_ACCEPTED")
            self.assertEqual(events[1]["event"], "duck")
        finally:
            asr.release.set()
            turn.release.set()
            await asyncio.gather(accept_task, return_exceptions=True)
            await runtime.flush()
            await runtime.close()

    async def test_turn_end_policy_receives_accumulated_asr_deltas(self):
        llm = _PromptRecordingLlm()
        runtime = ServerRealtimeSessionRuntime(
            "session-full-turn",
            llm=llm,
            tts=_SharedTtsRuntime(),
            asr=_FakeStreamingAsr(
                [
                    TranscriptChunk("chunk-1", "北京", 1.0, False, revision_id=1),
                    TranscriptChunk("chunk-2", "天气", 2.0, False, revision_id=2),
                ]
            ),
            turn=_SequencedTurn(),
            fusion=SpeechEventFusion(turn_end_frames=1),
            prompt_builder=lambda text: text,
        )
        for sequence in range(2):
            await runtime.accept_audio_frame(
                RealtimeAudioFrame(
                    AudioFrameHeader(sequence=sequence, capture_timestamp=float(sequence)),
                    b"\x00\x00" * 320,
                )
            )
        await runtime.flush()
        await asyncio.wait_for(llm.called.wait(), 0.2)

        self.assertEqual(llm.prompts, ["北京天气"])
        await runtime.close()

    async def test_turn_end_policy_applies_authoritative_asr_replacement(self):
        llm = _PromptRecordingLlm()
        runtime = ServerRealtimeSessionRuntime(
            "session-replaced-turn",
            llm=llm,
            tts=_SharedTtsRuntime(),
            asr=_FakeStreamingAsr(
                [
                    TranscriptChunk("chunk-1", "我说上海", 1.0, False, revision_id=1),
                    TranscriptChunk(
                        "chunk-2",
                        "",
                        2.0,
                        True,
                        revision_id=2,
                        committed_text="我说北京",
                        replaces_committed=True,
                    ),
                ]
            ),
            turn=_SequencedTurn(),
            fusion=SpeechEventFusion(turn_end_frames=1),
            prompt_builder=lambda text: text,
        )
        for sequence in range(2):
            await runtime.accept_audio_frame(
                RealtimeAudioFrame(
                    AudioFrameHeader(sequence=sequence, capture_timestamp=float(sequence)),
                    b"\x00\x00" * 320,
                )
            )
        await runtime.flush()
        await asyncio.wait_for(llm.called.wait(), 0.2)

        self.assertEqual(llm.prompts, ["我说北京"])
        await runtime.close()

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

    async def test_idle_cancel_blocks_new_operation_until_control_completes(self):
        runtime = _IdleControlRaceRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        stale = _SessionRuntimeProxy(boundary)
        fresh = _SessionRuntimeProxy(boundary)

        cancel_task = asyncio.create_task(stale.cancel("stale-request"))
        await runtime.control_started.wait()
        generate_task = asyncio.create_task(fresh.generate("fresh"))
        await asyncio.sleep(0)

        self.assertFalse(runtime.operation_started.is_set())
        self.assertEqual(runtime.cancel_calls, [("stale-request",)])

        runtime.allow_control_finish.set()
        cancel_result = await cancel_task
        generate_result = await generate_task

        self.assertTrue(cancel_result)
        self.assertEqual(generate_result, "generated:fresh")
        self.assertEqual(runtime.generate_calls, ["fresh"])

    async def test_idle_reset_blocks_new_operation_until_control_completes(self):
        runtime = _IdleControlRaceRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        stale = _SessionRuntimeProxy(boundary)
        fresh = _SessionRuntimeProxy(boundary)

        reset_task = asyncio.create_task(boundary.control(stale, "reset"))
        await runtime.control_started.wait()
        generate_task = asyncio.create_task(fresh.generate("fresh"))
        await asyncio.sleep(0)

        self.assertFalse(runtime.operation_started.is_set())
        self.assertEqual(runtime.reset_calls, 1)

        runtime.allow_control_finish.set()
        reset_result = await reset_task
        generate_result = await generate_task

        self.assertTrue(reset_result)
        self.assertEqual(generate_result, "generated:fresh")

    async def test_owner_cancel_still_forwards_while_stream_holds_operation_lock(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        owner = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in owner.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()

        cancel_task = asyncio.create_task(owner.cancel("owner-request"))
        await asyncio.sleep(0)

        self.assertEqual(runtime.cancel_calls, [("owner-request",)])
        await task
        self.assertTrue(await cancel_task)

    async def test_waiting_session_control_is_rejected_while_owner_active(self):
        runtime = _BlockingRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        owner = _SessionRuntimeProxy(boundary)
        waiting = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in owner.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.active.wait()

        self.assertFalse(await boundary.control(waiting, "reset"))
        self.assertEqual(runtime.reset_calls, 0)

        await owner.cancel()
        await task

    async def test_owner_is_cleared_after_control_exception(self):
        runtime = _ExplodingControlRuntime()
        boundary = _SharedRuntimeBoundary(runtime)
        owner = _SessionRuntimeProxy(boundary)
        fresh = _SessionRuntimeProxy(boundary)

        async def consume():
            return [chunk async for chunk in owner.stream_tokens("hello")]

        task = asyncio.create_task(consume())
        await runtime.operation_started.wait()

        with self.assertRaisesRegex(RuntimeError, "cancel boom"):
            await owner.cancel()
        await task

        result = await fresh.generate("fresh")
        self.assertEqual(result, "fresh")

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

    async def test_production_factory_reuses_chat_qwen_for_translation_by_default(self):
        loaded = []
        shared_llm = _SharedLlmRuntime()
        runtimes = {
            "asr": _SharedAsrRuntime(),
            "turn": _SharedTurnRuntime(),
            "policy": _SharedPolicyRuntime(),
            "llm": shared_llm,
            "tts": _SharedTtsRuntime(),
            "translation": _SharedTranslationRuntime(),
        }

        def loader(name):
            def load(_profile):
                loaded.append(name)
                return runtimes[name]
            return load

        factory = build_production_runtime_factory(
            model_config=MODEL_CONFIG,
            runtime_loaders={name: loader(name) for name in runtimes},
        )

        session = factory("session-shared-qwen")

        self.assertNotIn("translation", loaded)
        self.assertIs(
            session.interpretation_translator.runtime.runtime,
            session.llm.provider.runtime.runtime,
        )
        self.assertIs(session.llm.provider.runtime.runtime.runtime, shared_llm)

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

        self.assertEqual(events[0]["event"], "set_epoch")
        self.assertEqual(events[1]["event"], "AUDIO_FRAME_ACCEPTED")
        self.assertEqual(events[2]["event"], "transcript")
        self.assertEqual(events[3]["event"], "translation")
        self.assertEqual(events[3]["payload"]["translated_text"], "Hello")
        self.assertEqual(events[4]["event"], "audio_chunk")
        self.assertEqual(tts_runtime.requests[0][0], "Hello")
        checkpoint = runtime.checkpoints.get(runtime.current_response_id)
        self.assertEqual(checkpoint.segments[0].text, "Hello")
        self.assertEqual(checkpoint.segments[0].audio, b"\x00\x00" * 4)
        self.assertEqual(len(translator_runtime.prompts), 1)

    async def test_mode_switch_publishes_new_epoch_after_stopping_previous_response(self):
        runtime = ServerRealtimeSessionRuntime(
            "session-mode-boundary",
            llm=_SharedLlmRuntime(),
            tts=_SharedTtsRuntime(),
        )
        runtime.activate_response(runtime.next_response_id())

        effect = await runtime.apply_controller_actions(
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

        events = []
        while not runtime._events.empty():
            events.append(runtime._events.get_nowait())

        self.assertEqual(effect.advanced_epoch, 1)
        self.assertEqual([item["event"] for item in events], ["stop_response", "set_epoch"])
        self.assertEqual(events[0]["payload"]["response_id"], "response-1")
        self.assertEqual(events[0]["payload"]["generation_epoch"], 0)
        self.assertEqual(events[1]["response_id"], "response-2")
        self.assertEqual(events[1]["generation_epoch"], 1)

        await runtime.close()

    async def test_stop_interpretation_control_is_classified_before_translation(self):
        translation = _SharedTranslationRuntime("should not be emitted")
        policy = _RecordingModePolicy(
            PolicyDecision(
                action=PolicyAction.MODE_SWITCH,
                confidence=0.98,
                rationale="explicitly stop continuous interpretation",
                intent="chat",
            )
        )
        runtime = ServerRealtimeSessionRuntime(
            "session-stop-interpretation",
            llm=_SharedLlmRuntime(),
            tts=_SharedTtsRuntime(),
            policy_engine=policy,
            interpretation_translator=type(
                "Translator",
                (),
                {
                    "__init__": lambda self, provider: setattr(self, "provider", provider),
                    "translate_stream": lambda self, prompt: self.provider.generate(prompt),
                },
            )(translation),
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

        await runtime._accept_transcript_update(
            TranscriptChunk("control-1", "停止同传", 1.0, True, revision_id=1)
        )

        self.assertEqual(runtime.conversation_mode.value, "CHAT")
        self.assertIsNone(runtime.current_response_id)
        self.assertEqual(translation.prompts, [])
        self.assertEqual(policy.requests[0].user_transcript.text, "停止同传")
        epoch_after_control = runtime.generation_epoch

        await runtime._apply_policy(
            SpeechCandidateEvent(
                event="USER_TURN_END_CANDIDATE",
                event_id="turn-end-control-1",
                timestamp=1.1,
                source="turn",
                payload={"label": "turn_end", "confidence": 1.0},
            ),
            await runtime._turn_transcript_snapshot(),
        )

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(runtime.generation_epoch, epoch_after_control)

        await runtime.close()

    async def test_change_interpretation_language_control_rotates_response_without_translation(self):
        translation = _SharedTranslationRuntime("should not be emitted")
        policy = _RecordingModePolicy(
            PolicyDecision(
                action=PolicyAction.MODE_SWITCH,
                confidence=0.97,
                rationale="change persistent target language",
                intent="continuous_interpretation",
                source_language="Chinese",
                target_language="Japanese",
            )
        )
        runtime = ServerRealtimeSessionRuntime(
            "session-change-language",
            llm=_SharedLlmRuntime(),
            tts=_SharedTtsRuntime(),
            policy_engine=policy,
            interpretation_translator=type(
                "Translator",
                (),
                {
                    "__init__": lambda self, provider: setattr(self, "provider", provider),
                    "translate_stream": lambda self, prompt: self.provider.generate(prompt),
                },
            )(translation),
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
        old_response_id = runtime.current_response_id

        await runtime._accept_transcript_update(
            TranscriptChunk("control-2", "接下来改成日语", 2.0, True, revision_id=1)
        )

        self.assertEqual(runtime.conversation_mode.value, "INTERPRETATION")
        self.assertEqual(runtime.interpretation_session.target_language, "Japanese")
        self.assertNotEqual(runtime.current_response_id, old_response_id)
        self.assertEqual(translation.prompts, [])

        await runtime.close()

    async def test_revise_generation_prompt_includes_original_request_and_previous_answer(self):
        llm = _ContextPromptLlm(("上海是一座城市。", "北京是中国的首都。"))
        runtime = ServerRealtimeSessionRuntime(
            "session-contextual-revise",
            llm=llm,
            tts=_SharedTtsRuntime(),
        )

        await runtime._start_generation("请介绍上海")
        await runtime._generation_slot.task
        await runtime._start_generation(
            "不对，我说的是北京，不是上海",
            policy_action=PolicyAction.REVISE,
        )
        await runtime._generation_slot.task

        self.assertIn("请介绍上海", llm.prompts[1])
        self.assertIn("上海是一座城市。", llm.prompts[1])
        self.assertIn("不对，我说的是北京，不是上海", llm.prompts[1])

        await runtime.close()

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

    def test_model_manager_uses_explicit_or_cuda_device_budget(self):
        """Catches production silently disabling VRAM admission without an explicit value."""
        cuda_config = {"models": {"llm": {"device": "cuda:1"}}}
        explicit_config = {
            "models": {"llm": {"device": "cuda:1"}},
            "runtime": {"vram_budget_bytes": 900, "vram_reserve_bytes": 100},
        }

        with patch(
            "src.runtime_app.container._cuda_device_total_memory_bytes",
            return_value=700,
        ):
            derived = _build_model_manager(cuda_config)
            explicit = _build_model_manager(explicit_config)

        self.assertEqual(derived.total_vram_bytes, 700)
        self.assertEqual(explicit.total_vram_bytes, 900)
        self.assertEqual(explicit.reserve_bytes, 100)

    def test_cuda_measurement_uses_device_wide_free_memory_delta_source(self):
        """Catches reverting to parent-process allocation for subprocess-backed models."""
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda device: (30, 100),
            )
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            used_bytes = _measure_cuda_allocated_bytes("cuda:1")

        self.assertEqual(used_bytes, 70)

    async def test_measured_overage_closes_partial_resource_and_releases_budget(self):
        """Catches a measured-over-budget model remaining live or reserved after rejection."""
        loaded = _ClosableLoadedRuntime()
        config = {
            "models": {
                "asr": {
                    "provider": "local",
                    "model_path": "/models/asr",
                    "device": "cuda",
                    "estimated_vram_bytes": 1,
                }
            },
            "runtime": {"vram_budget_bytes": 50},
        }
        with patch(
            "src.runtime_app.container._measure_cuda_allocated_bytes",
            side_effect=(0, 60),
        ):
            factory = build_production_runtime_factory(
                model_config=config,
                runtime_loaders={"asr": lambda _profile: loaded},
            )
            with self.assertRaises(ModelMemoryBudgetError):
                factory("session-over-budget")

        self.assertTrue(loaded.closed)
        self.assertEqual(factory.memory_diagnostics["loaded_bytes"], 0)
        self.assertEqual(factory.memory_diagnostics["reserved_bytes"], 0)

    async def test_production_tts_recovery_reloads_and_replaces_shared_worker(self):
        """Catches production recovery resetting the same closed shared worker."""
        failed = _ReloadableTtsRuntime(fail=True)
        replacement = _ReloadableTtsRuntime()
        tts_runtimes = iter((failed, replacement))
        shared = {
            "asr": _SharedAsrRuntime(),
            "turn": _SharedTurnRuntime(),
            "policy": _SharedPolicyRuntime(),
            "llm": _SharedLlmRuntime(),
            "translation": _SharedTranslationRuntime(),
        }
        loaders = {
            name: (lambda _config, runtime=runtime: runtime)
            for name, runtime in shared.items()
        }
        loaders["tts"] = lambda _profile: next(tts_runtimes)
        factory = build_production_runtime_factory(
            model_config=MODEL_CONFIG,
            runtime_loaders=loaders,
        )
        runtime = factory("session-reload-tts")
        runtime.activate_response("response-reload")
        runtime.checkpoints.record_segment("response-reload", 0, "checkpoint replay")
        segment = TextSegment(0, "caller text", False, "response-reload", 0)

        keep_running = await runtime._stream_tts_segment(segment, 0, CancellationToken())

        self.assertTrue(keep_running)
        self.assertTrue(failed.closed)
        self.assertEqual([item[0] for item in failed.requests], ["caller text"])
        self.assertEqual([item[0] for item in replacement.requests], ["checkpoint replay"])
        await runtime.close()
        await factory.close()

    async def test_factory_close_attempts_all_resources_and_releases_all_reservations(self):
        """Catches one close failure preventing later cleanup and accounting release."""
        closed = []
        runtimes = {
            name: _CloseRecordingRuntime(
                name,
                closed,
                error=RuntimeError("asr close failed") if name == "asr" else None,
            )
            for name in MODEL_CONFIG["models"]
        }
        factory = build_production_runtime_factory(
            model_config={
                "models": MODEL_CONFIG["models"],
                "runtime": {"vram_budget_bytes": 1000},
            },
            runtime_loaders={
                name: (lambda _profile, runtime=runtime: runtime)
                for name, runtime in runtimes.items()
            },
        )
        factory("session-close-all")

        with self.assertRaisesRegex(RuntimeError, "asr close failed"):
            await factory.close()

        self.assertEqual(set(closed), set(MODEL_CONFIG["models"]) - {"translation"})
        self.assertEqual(factory.memory_diagnostics["models"], {})
        self.assertEqual(factory.memory_diagnostics["loaded_bytes"], 0)
        self.assertEqual(factory.memory_diagnostics["reserved_bytes"], 0)
