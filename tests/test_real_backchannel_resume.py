import unittest

from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.playback import PlaybackCoordinator
from src.realtime.session_runtime import RealtimeSessionRuntime
from src.realtime.session_state import ConversationMode, FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent
from src.realtime.text_segmenter import TextSegment
from src.tts_runtime.stream import AudioChunk


def candidate(event, label):
    return SpeechCandidateEvent(
        event=event,
        event_id="candidate-1",
        timestamp=1.0,
        source="test",
        payload={"label": label, "confidence": 0.9, "evidence": {}},
    )


def decision(action):
    return PolicyDecision(action=action, confidence=0.97, rationale="test")


class RealBackchannelResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_backchannel_restores_without_advancing_epoch_or_archiving_response(self):
        commands = []
        runtime = RealtimeSessionRuntime(
            "session-1",
            playback=PlaybackCoordinator(send_command=commands.append),
        )
        runtime.activate_response("response-1")
        controller = ConversationController()
        state = SessionState(
            mode=ConversationMode.CHAT,
            floor=FloorState.ASSISTANT,
            response=ResponseState.PLAYING,
            assistant_act="EXPLAINING",
        )

        tentative = controller.handle_candidate(
            state, candidate("USER_SPEECH_START_CANDIDATE", "speaking")
        )
        await runtime.apply_controller_actions(tentative)
        effect = await runtime.apply_controller_actions(
            controller.apply_policy(state, decision(PolicyAction.BACKCHANNEL))
        )

        self.assertEqual(runtime.generation_epoch, 0)
        self.assertEqual(runtime.current_response_id, "response-1")
        self.assertEqual(runtime.archived_response_ids, ())
        self.assertIsNone(effect.archived_response_id)
        self.assertIsNone(effect.advanced_epoch)
        self.assertEqual(commands, ["DUCK", "RESTORE"])

    async def test_pause_then_resume_replays_the_partial_phrase_from_offset_zero(self):
        commands = []
        runtime = RealtimeSessionRuntime(
            "session-2",
            playback=PlaybackCoordinator(send_command=commands.append),
        )
        runtime.activate_response("response-2")
        runtime.record_generated_text("response-2", "第二个是故宫。")
        runtime.record_response_segment(
            TextSegment(
                segment_id=0,
                text="第二个是故宫。",
                is_final=False,
                response_id="response-2",
                generation_epoch=0,
            )
        )
        runtime.record_audio_chunk(
            AudioChunk(
                chunk_id="chunk-1",
                audio_data=b"\x00\x00" * 6,
                timestamp=1.0,
                is_final=False,
                request_id="response-2:0:0",
                response_id="response-2",
                generation_epoch=0,
                segment_id=0,
            )
        )
        runtime.ack_playback("response-2", generation_epoch=0, segment_id=0, sample_offset=5)

        await runtime.apply_controller_actions((ControllerAction(ActionType.PAUSE_RESPONSE),))
        effect = await runtime.apply_controller_actions((ControllerAction(ActionType.RESUME_RESPONSE),))

        self.assertEqual(effect.resume_plan.segment_id, 0)
        self.assertEqual(effect.resume_plan.sample_offset, 0)
        self.assertEqual(effect.resume_plan.audio, b"\x00\x00" * 6)
        self.assertEqual(commands, ["PAUSE", "RESUME"])

    async def test_revise_advances_epoch_discards_unplayed_audio_and_new_request_archives_old_response(self):
        commands = []
        runtime = RealtimeSessionRuntime(
            "session-3",
            playback=PlaybackCoordinator(send_command=commands.append),
        )
        runtime.activate_response("response-3")
        runtime.record_generated_text("response-3", "先说上海，再说北京。")
        runtime.record_response_segment(
            TextSegment(
                segment_id=0,
                text="先说上海。",
                is_final=False,
                response_id="response-3",
                generation_epoch=0,
            )
        )
        runtime.record_audio_chunk(
            AudioChunk(
                chunk_id="chunk-1",
                audio_data=b"\x00\x00" * 4,
                timestamp=1.0,
                is_final=False,
                request_id="response-3:0:0",
                response_id="response-3",
                generation_epoch=0,
                segment_id=0,
            )
        )
        runtime.record_response_segment(
            TextSegment(
                segment_id=1,
                text="再说北京。",
                is_final=False,
                response_id="response-3",
                generation_epoch=0,
            )
        )
        runtime.record_audio_chunk(
            AudioChunk(
                chunk_id="chunk-2",
                audio_data=b"\x00\x00" * 4,
                timestamp=1.1,
                is_final=False,
                request_id="response-3:0:1",
                response_id="response-3",
                generation_epoch=0,
                segment_id=1,
            )
        )
        runtime.ack_playback("response-3", generation_epoch=0, segment_id=0, sample_offset=4)

        revise = await runtime.apply_controller_actions(
            (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
                ControllerAction(ActionType.REVISE_RESPONSE),
                ControllerAction(ActionType.PROCESS_USER_REQUEST),
            )
        )
        stale_kept = runtime.record_audio_chunk(
            AudioChunk(
                chunk_id="late-old",
                audio_data=b"\x00\x00" * 4,
                timestamp=2.0,
                is_final=False,
                request_id="response-3:0:1",
                response_id="response-3",
                generation_epoch=0,
                segment_id=1,
            )
        )

        runtime.activate_response("response-4")
        new_request = await runtime.apply_controller_actions(
            (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
                ControllerAction(ActionType.PROCESS_USER_REQUEST),
            )
        )

        revised = runtime.checkpoints.get("response-3")
        self.assertEqual(revise.advanced_epoch, 1)
        self.assertIsNone(revise.archived_response_id)
        self.assertFalse(stale_kept)
        self.assertFalse(revised.archived)
        self.assertFalse(revised.resumable)
        self.assertIsNone(runtime.checkpoints.resume("response-3"))
        self.assertEqual(new_request.advanced_epoch, 2)
        self.assertEqual(new_request.archived_response_id, "response-4")
        self.assertEqual(runtime.archived_response_ids, ("response-4",))
        self.assertEqual(commands, ["STOP", "STOP"])


if __name__ == "__main__":
    unittest.main()
