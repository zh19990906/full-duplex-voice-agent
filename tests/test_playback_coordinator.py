import unittest

from src.api.events import ApiEventSerializer
from src.controller.actions import ActionType, ControllerAction
from src.realtime.checkpoint import ResponseCheckpointStore
from src.realtime.playback import PlaybackAck, PlaybackCoordinator


class PlaybackCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_playback_commands_update_checkpoint_and_ignore_stale_ack_after_epoch_change(self):
        commands = []
        store = ResponseCheckpointStore()
        store.activate("response-1", generation_epoch=0)
        store.record_segment("response-1", 0, "hello", audio=b"\x00\x00" * 8)
        playback = PlaybackCoordinator(send_command=commands.append, checkpoint_store=store)
        playback.set_active_response("response-1", generation_epoch=0)

        await playback.apply(ControllerAction(ActionType.DUCK_RESPONSE))
        await playback.apply(ControllerAction(ActionType.RESTORE_RESPONSE))
        await playback.apply(ControllerAction(ActionType.PAUSE_RESPONSE))

        accepted = playback.ack(
            PlaybackAck(
                response_id="response-1",
                generation_epoch=0,
                segment_id=0,
                sample_offset=4,
                audio_time=1.25,
            )
        )
        playback.set_active_response("response-1", generation_epoch=1)
        stale = playback.ack(
            PlaybackAck(
                response_id="response-1",
                generation_epoch=0,
                segment_id=0,
                sample_offset=8,
                audio_time=1.50,
            )
        )

        checkpoint = store.get("response-1")
        self.assertTrue(accepted)
        self.assertFalse(stale)
        self.assertEqual(checkpoint.played_cursor, 4)
        self.assertEqual(commands, ["DUCK", "RESTORE", "PAUSE"])

    async def test_resume_action_replays_partially_played_segment_from_offset_zero(self):
        commands = []
        store = ResponseCheckpointStore()
        store.activate("response-2", generation_epoch=2)
        store.record_segment("response-2", 3, "继续说完这一句。", audio=b"\x00\x00" * 6)
        store.ack("response-2", segment_id=3, sample_offset=5)
        store.pause("response-2")
        playback = PlaybackCoordinator(send_command=commands.append, checkpoint_store=store)
        playback.set_active_response("response-2", generation_epoch=2)

        plan = await playback.apply(ControllerAction(ActionType.RESUME_RESPONSE))

        self.assertEqual(plan.segment_id, 3)
        self.assertEqual(plan.sample_offset, 0)
        self.assertEqual(plan.audio, b"\x00\x00" * 6)
        self.assertEqual(
            commands,
            [
                {
                    "event": "RESUME_RESPONSE",
                    "payload": {
                        "response_id": "response-2",
                        "generation_epoch": 2,
                        "playback_attempt_id": plan.playback_attempt_id,
                    },
                }
            ],
        )

        serialized = ApiEventSerializer.serialize(commands[0], "session-2")
        self.assertEqual(
            serialized,
            {
                "event": "RESUME_RESPONSE",
                "response_id": "response-2",
                "generation_epoch": 2,
                "playback_attempt_id": plan.playback_attempt_id,
                "payload": {
                    "response_id": "response-2",
                    "generation_epoch": 2,
                    "playback_attempt_id": plan.playback_attempt_id,
                },
                "session_id": "session-2",
            },
        )

    async def test_resume_rejects_delayed_ack_from_old_playback_attempt(self):
        commands = []
        store = ResponseCheckpointStore()
        store.activate("response-3", generation_epoch=2)
        store.record_segment("response-3", 3, "继续说完这一句。", audio=b"\x00\x00" * 6)
        playback = PlaybackCoordinator(send_command=commands.append, checkpoint_store=store)
        playback.set_active_response("response-3", generation_epoch=2)
        first_attempt = playback.active_playback_attempt_id

        await playback.apply(ControllerAction(ActionType.PAUSE_RESPONSE))
        plan = await playback.apply(ControllerAction(ActionType.RESUME_RESPONSE))

        stale_attempt = playback.ack(
            PlaybackAck(
                response_id="response-3",
                generation_epoch=2,
                segment_id=3,
                sample_offset=4,
                audio_time=1.0,
                playback_attempt_id=first_attempt,
            )
        )
        missing_attempt = playback.ack(
            PlaybackAck(
                response_id="response-3",
                generation_epoch=2,
                segment_id=3,
                sample_offset=4,
                audio_time=1.1,
            )
        )
        accepted = playback.ack(
            PlaybackAck(
                response_id="response-3",
                generation_epoch=2,
                segment_id=3,
                sample_offset=5,
                audio_time=1.2,
                playback_attempt_id=plan.playback_attempt_id,
            )
        )

        checkpoint = store.get("response-3")
        self.assertFalse(stale_attempt)
        self.assertFalse(missing_attempt)
        self.assertTrue(accepted)
        self.assertEqual(checkpoint.played_cursor, 5)
        self.assertEqual(plan.playback_attempt_id, first_attempt + 1)


if __name__ == "__main__":
    unittest.main()
