import asyncio
import unittest

from src.adapters.turn.x2_turn_streaming import TurnCandidate
from src.asr.stream import TranscriptChunk
from src.realtime.audio_ingress import AudioActivityCandidate, RealtimeAudioFrame
from src.realtime.cancellation import CancellationToken
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.protocol import AudioFrameHeader
from src.realtime.speech_fusion import SpeechEventFusion
from src.realtime.text_segmenter import TextSegment
from src.runtime_app.container import ServerRealtimeSessionRuntime
from src.tts_runtime.stream import AudioChunk


class _IdleLlm:
    async def stream_tokens(self, _prompt, **_options):
        if False:
            yield None

    def reset(self):
        return None

    async def interrupt(self):
        return None


class _TrackingTts:
    def __init__(self):
        self.interrupt_calls = 0
        self.close_calls = 0

    async def stream_audio(self, _text, **_options):
        if False:
            yield b""

    async def interrupt(self, *_args, **_kwargs):
        self.interrupt_calls += 1

    async def close(self):
        self.close_calls += 1

    def reset(self):
        return None


class _ExplodingLlm(_IdleLlm):
    async def stream_tokens(self, _prompt, **_options):
        yield "preserve this"
        raise RuntimeError("llm broke")


class _SingleSegmentLlm(_IdleLlm):
    async def stream_tokens(self, _prompt, **_options):
        yield "preserve this."


class _FinalFailingAsr:
    async def push_pcm(self, _frame):
        return TranscriptChunk("partial", "stable request", 1.0, False, revision_id=1)

    async def finalize_turn(self):
        raise RuntimeError("final ASR broke")


class _TurnEnd:
    async def push_pcm(self, _frame):
        return (TurnCandidate("turn_end"),)


class _RecordingPolicy:
    def __init__(self):
        self.requests = []

    async def decide(self, request):
        self.requests.append(request)
        return PolicyDecision(PolicyAction.PAUSE, 1.0, "test")


class _FailingTts(_TrackingTts):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def stream_audio(self, text, **_options):
        self.requests.append(text)
        raise RuntimeError("worker died")
        if False:
            yield b""


class _WorkingTts(_TrackingTts):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def stream_audio(self, text, **_options):
        self.requests.append(text)
        yield b"\x00\x00" * 2


class RealtimeDegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_policy_timeout_safely_pauses_and_requests_clarification(self):
        """Catches UNCERTAIN policy fallback that keeps talking instead of pausing."""
        runtime = ServerRealtimeSessionRuntime("session-policy", llm=_IdleLlm(), tts=_TrackingTts())

        await runtime.handle_policy_timeout()
        first = await asyncio.wait_for(anext(runtime.events()), timeout=0.1)
        second = await asyncio.wait_for(anext(runtime.events()), timeout=0.1)

        self.assertEqual(runtime.session_state.response.value, "PAUSED")
        self.assertEqual(runtime.session_state.floor.value, "USER")
        self.assertEqual(first["event"], "policy_uncertain")
        self.assertEqual(second["event"], "request_clarification")

    async def test_asr_failure_prompts_the_user_to_repeat(self):
        """Catches ASR failure paths that keep making semantic decisions on bad input."""
        runtime = ServerRealtimeSessionRuntime("session-asr", llm=_IdleLlm(), tts=_TrackingTts())

        await runtime.handle_asr_failure(RuntimeError("asr broke"))
        event = await asyncio.wait_for(anext(runtime.events()), timeout=0.1)

        self.assertEqual(event["event"], "request_repeat")
        self.assertIn("asr broke", event["payload"]["message"])

    async def test_playback_failure_stops_later_audio_enqueue(self):
        """Catches playback faults that keep pushing audio and growing queues."""
        runtime = ServerRealtimeSessionRuntime("session-playback", llm=_IdleLlm(), tts=_TrackingTts())
        runtime.activate_response("response-4")

        await runtime.handle_playback_failure(RuntimeError("speaker lost"))
        failure_event = await asyncio.wait_for(anext(runtime.events()), timeout=0.1)
        await runtime._publish_audio_chunk(
            AudioChunk(
                chunk_id="chunk-1",
                audio_data=b"\x00\x00" * 2,
                timestamp=1.0,
                is_final=False,
                request_id="response-4:0:0",
                response_id="response-4",
                generation_epoch=runtime.generation_epoch,
                segment_id=0,
            )
        )

        self.assertEqual(failure_event["event"], "playback_failed")
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(runtime.events()), timeout=0.05)

    async def test_cancel_and_close_are_idempotent(self):
        """Catches repeated interrupt/close calls that raise or duplicate teardown badly."""
        tts = _TrackingTts()
        runtime = ServerRealtimeSessionRuntime("session-close", llm=_IdleLlm(), tts=tts)

        await runtime.accept_command({"type": "interrupt"})
        await runtime.accept_command({"type": "interrupt"})
        await runtime.close()
        await runtime.close()

        self.assertTrue(runtime.closed)
        self.assertGreaterEqual(tts.interrupt_calls, 2)

    async def test_final_asr_failure_aborts_policy_and_generation(self):
        """Catches policy running on a stale partial when final ASR fails."""
        policy = _RecordingPolicy()
        runtime = ServerRealtimeSessionRuntime(
            "session-final-asr",
            llm=_IdleLlm(),
            tts=_TrackingTts(),
            asr=_FinalFailingAsr(),
            turn=_TurnEnd(),
            policy_engine=policy,
            fusion=SpeechEventFusion(turn_end_frames=1),
        )
        frame = RealtimeAudioFrame(
            AudioFrameHeader(sequence=0, capture_timestamp=1.0),
            b"\x00\x00" * 320,
        )

        await runtime.accept_audio_frame(frame)

        self.assertEqual(policy.requests, [])
        self.assertEqual(runtime._response_identifiers, 0)
        await runtime.close()

    async def test_x2_timeout_requires_silence_and_stable_or_final_asr(self):
        """Catches timeout fallback ending a turn during speech or on unstable-only ASR."""
        policy = _RecordingPolicy()
        runtime = ServerRealtimeSessionRuntime(
            "session-turn-timeout",
            llm=_IdleLlm(),
            tts=_TrackingTts(),
            policy_engine=policy,
        )
        frame = RealtimeAudioFrame(
            AudioFrameHeader(sequence=4, capture_timestamp=2.0),
            b"\x00\x00" * 320,
        )
        stable = TranscriptChunk("stable", "committed", 2.0, False, revision_id=1)
        unstable = TranscriptChunk(
            "unstable", "", 2.0, False, revision_id=2, unstable_text="maybe"
        )
        final = TranscriptChunk("final", "done", 2.0, True, revision_id=3)

        runtime.ingress._last_activity_candidate = AudioActivityCandidate(4, 2.0, True, 900.0)
        await runtime._handle_turn_timeout(stable, frame)
        runtime.ingress._last_activity_candidate = AudioActivityCandidate(4, 2.0, False, 0.0)
        await runtime._handle_turn_timeout(unstable, frame)
        await runtime._handle_turn_timeout(stable, frame)
        await runtime._handle_turn_timeout(final, frame)

        self.assertEqual(len(policy.requests), 2)
        await runtime.close()

    async def test_llm_failure_pauses_checkpoint_and_publishes_identity(self):
        """Catches LLM failure dropping resumable text or emitting an uncorrelated event."""
        runtime = ServerRealtimeSessionRuntime(
            "session-llm-failure", llm=_ExplodingLlm(), tts=_TrackingTts()
        )

        await runtime._run_generation("hello")
        events = [await asyncio.wait_for(anext(runtime.events()), 0.2) for _ in range(3)]
        failure = events[-1]
        checkpoint = runtime.checkpoints.get(failure["response_id"])

        self.assertEqual(failure["event"], "llm_failed")
        self.assertEqual(failure["payload"]["response_id"], failure["response_id"])
        self.assertEqual(
            failure["payload"]["generation_epoch"], failure["generation_epoch"]
        )
        self.assertTrue(checkpoint.paused)
        self.assertEqual(checkpoint.generated_text, "preserve this")
        self.assertEqual(runtime.session_state.response.value, "PAUSED")
        await runtime.close()

    async def test_tts_recovery_replaces_worker_and_resubmits_checkpoint_text_once(self):
        """Catches reset of the same closed worker or retrying non-checkpoint text."""
        failed = _FailingTts()
        replacement = _WorkingTts()

        async def recover():
            await failed.close()
            return replacement

        runtime = ServerRealtimeSessionRuntime(
            "session-tts-recovery",
            llm=_IdleLlm(),
            tts=failed,
            tts_recovery=recover,
        )
        runtime.activate_response("response-recovery")
        runtime.checkpoints.record_segment("response-recovery", 0, "checkpoint text")
        segment = TextSegment(0, "caller text", False, "response-recovery", 0)

        keep_running = await runtime._stream_tts_segment(
            segment, 0, CancellationToken()
        )

        self.assertTrue(keep_running)
        self.assertEqual(failed.requests, ["caller text"])
        self.assertEqual(failed.close_calls, 1)
        self.assertEqual(replacement.requests, ["checkpoint text"])
        await runtime.close()

    async def test_tts_recovery_is_fenced_when_generation_becomes_stale(self):
        """Catches a replacement worker synthesizing text after its epoch is invalidated."""
        failed = _FailingTts()
        replacement = _WorkingTts()
        recovery_started = asyncio.Event()
        allow_recovery = asyncio.Event()

        async def recover():
            recovery_started.set()
            await allow_recovery.wait()
            return replacement

        runtime = ServerRealtimeSessionRuntime(
            "session-tts-stale", llm=_IdleLlm(), tts=failed, tts_recovery=recover
        )
        runtime.activate_response("response-stale")
        runtime.checkpoints.record_segment("response-stale", 0, "do not replay")
        segment = TextSegment(0, "do not replay", False, "response-stale", 0)
        task = asyncio.create_task(
            runtime._stream_tts_segment(segment, 0, CancellationToken())
        )
        await asyncio.wait_for(recovery_started.wait(), 0.5)

        runtime.advance_generation()
        allow_recovery.set()

        self.assertFalse(await task)
        self.assertEqual(replacement.requests, [])
        await runtime.close()

    async def test_tts_reload_failure_stays_terminal_tts_path_with_checkpoint(self):
        """Catches worker recreation failure bubbling out as an LLM failure."""
        failed = _FailingTts()

        async def fail_recovery():
            raise RuntimeError("tts reload failed")

        runtime = ServerRealtimeSessionRuntime(
            "session-tts-reload-failure",
            llm=_SingleSegmentLlm(),
            tts=failed,
            tts_recovery=fail_recovery,
        )

        await runtime._run_generation("hello")
        events = []
        while not runtime._events.empty():
            events.append(runtime._events.get_nowait())
        named = [event for event in events if isinstance(event, dict)]
        names = [event.get("event") for event in named]
        failures = [
            event for event in named if event.get("event") in {"tts_failed", "worker_terminal"}
        ]
        checkpoint = runtime.checkpoints.get(runtime.current_response_id)

        self.assertIn("tts_failed", names)
        self.assertIn("worker_terminal", names)
        self.assertNotIn("llm_failed", names)
        self.assertNotIn("response_completed", names)
        self.assertTrue(checkpoint.paused)
        self.assertEqual(checkpoint.segments[0].text, "preserve this.")
        for failure in failures:
            self.assertEqual(failure["response_id"], runtime.current_response_id)
            self.assertEqual(failure["generation_epoch"], runtime.generation_epoch)
            self.assertEqual(failure["payload"]["response_id"], runtime.current_response_id)
            self.assertEqual(failure["payload"]["generation_epoch"], runtime.generation_epoch)
        await runtime.close()


if __name__ == "__main__":
    unittest.main()
