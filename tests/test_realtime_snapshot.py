import asyncio
import unittest

from src.controller.actions import ActionType, ControllerAction
from src.realtime.session_state import ConversationMode, FloorState, ResponseState
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


class _IdleTts:
    async def stream_audio(self, _text, **_options):
        if False:
            yield b""

    def reset(self):
        return None

    async def interrupt(self, *_args, **_kwargs):
        return None


class RealtimeSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_publishes_reconnect_snapshot_with_playback_state(self):
        """Catches reconnects that lose mode, response, or playback cursor state."""
        runtime = ServerRealtimeSessionRuntime("session-snapshot", llm=_IdleLlm(), tts=_IdleTts())
        response_id = "response-7"
        runtime.session_state.mode = ConversationMode.INTERPRETATION
        runtime.conversation_mode = ConversationMode.INTERPRETATION
        runtime.session_state.floor = FloorState.USER
        runtime.activate_response(response_id)
        runtime.record_generated_text(response_id, "继续讲北京。")
        runtime.record_response_segment(
            TextSegment(
                segment_id=0,
                text="继续讲北京。",
                is_final=False,
                response_id=response_id,
                generation_epoch=runtime.generation_epoch,
            )
        )
        runtime.record_audio_chunk(
            AudioChunk(
                chunk_id="chunk-1",
                audio_data=b"\x00\x00" * 6,
                timestamp=1.0,
                is_final=False,
                request_id=f"{response_id}:{runtime.generation_epoch}:0",
                response_id=response_id,
                generation_epoch=runtime.generation_epoch,
                segment_id=0,
            )
        )
        runtime.ack_playback(
            response_id,
            generation_epoch=runtime.generation_epoch,
            segment_id=0,
            sample_offset=4,
        )
        runtime.session_state.response = ResponseState.PAUSED
        runtime.checkpoints.pause(response_id)

        await runtime.connect()
        event = await asyncio.wait_for(anext(runtime.events()), timeout=0.1)

        self.assertEqual(event["event"], "session_snapshot")
        self.assertEqual(event["payload"]["mode"], "INTERPRETATION")
        self.assertEqual(event["payload"]["floor"], "USER")
        self.assertEqual(event["payload"]["response"], "PAUSED")
        self.assertEqual(event["payload"]["generation_epoch"], runtime.generation_epoch)
        self.assertEqual(event["payload"]["current_response_id"], response_id)
        self.assertEqual(event["payload"]["playback_cursor"]["played_cursor"], 4)
        self.assertEqual(event["payload"]["paused_responses"][0]["response_id"], response_id)
        self.assertEqual(event["payload"]["paused_responses"][0]["text"], "继续讲北京。")


if __name__ == "__main__":
    unittest.main()
