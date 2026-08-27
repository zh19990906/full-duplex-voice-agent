import asyncio
import unittest

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


if __name__ == "__main__":
    unittest.main()
