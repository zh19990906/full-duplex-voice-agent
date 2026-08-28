import unittest

from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.playback import PlaybackCoordinator
from src.realtime.session_runtime import RealtimeSessionRuntime
from src.realtime.session_state import ConversationMode, FloorState, ResponseState, SessionState


class InterpretationModePolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_one_shot_translation_does_not_switch_mode(self):
        state = SessionState(
            mode=ConversationMode.CHAT,
            floor=FloorState.USER,
            response=ResponseState.IDLE,
        )
        controller = ConversationController()
        policy = PolicyDecision(
            action=PolicyAction.ANSWER,
            confidence=0.99,
            intent="translate_once",
            rationale="A single translation request is ordinary chat.",
        )

        actions = controller.apply_policy(state, policy)

        self.assertEqual(state.mode, ConversationMode.CHAT)
        self.assertEqual(
            [item.action_type for item in actions],
            [ActionType.STOP_RESPONSE, ActionType.PROCESS_USER_REQUEST],
        )

    async def test_runtime_enters_and_exits_interpretation_only_on_switch_action(self):
        runtime = RealtimeSessionRuntime("session-interpret")

        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )
        self.assertEqual(runtime.conversation_mode, ConversationMode.INTERPRETATION)
        self.assertIsNotNone(runtime.interpretation_session)
        self.assertEqual(runtime.interpretation_session.target_language, "English")

        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "French",
                    },
                ),
            )
        )
        self.assertEqual(runtime.conversation_mode, ConversationMode.INTERPRETATION)
        self.assertEqual(runtime.interpretation_session.target_language, "French")

        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "CHAT",
                        "source_language": None,
                        "target_language": None,
                    },
                ),
            )
        )
        self.assertEqual(runtime.conversation_mode, ConversationMode.CHAT)
        self.assertIsNone(runtime.interpretation_session)

    async def test_mode_switch_is_atomic_and_uses_one_session_response_identity_sequence(self):
        playback_commands = []
        cancelled = []

        async def cancel_generation(response_id, generation_epoch):
            cancelled.append((response_id, generation_epoch))

        runtime = RealtimeSessionRuntime(
            "session-atomic-mode",
            playback=PlaybackCoordinator(send_command=playback_commands.append),
            cancel_generation=cancel_generation,
        )
        chat_response_id = runtime.next_response_id()
        runtime.activate_response(chat_response_id)

        first_effect = await runtime.apply_controller_actions(
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

        self.assertEqual(cancelled, [("response-1", 0)])
        self.assertEqual(playback_commands, ["STOP"])
        self.assertEqual(first_effect.advanced_epoch, 1)
        self.assertEqual(runtime.current_response_id, "response-2")
        self.assertFalse(runtime.checkpoints.get("response-1").resumable)

        second_effect = await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": "Chinese",
                        "target_language": "French",
                    },
                ),
            )
        )

        self.assertEqual(cancelled[-1], ("response-2", 1))
        self.assertEqual(playback_commands, ["STOP", "STOP"])
        self.assertEqual(second_effect.advanced_epoch, 2)
        self.assertEqual(runtime.current_response_id, "response-3")
        self.assertEqual(runtime.interpretation_session.target_language, "French")
        self.assertFalse(runtime.checkpoints.get("response-2").resumable)

        third_effect = await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "CHAT",
                        "source_language": None,
                        "target_language": None,
                    },
                ),
            )
        )

        self.assertEqual(cancelled[-1], ("response-3", 2))
        self.assertEqual(playback_commands, ["STOP", "STOP", "STOP"])
        self.assertEqual(third_effect.advanced_epoch, 3)
        self.assertEqual(runtime.conversation_mode, ConversationMode.CHAT)
        self.assertIsNone(runtime.current_response_id)
        self.assertFalse(runtime.checkpoints.get("response-3").resumable)


if __name__ == "__main__":
    unittest.main()
