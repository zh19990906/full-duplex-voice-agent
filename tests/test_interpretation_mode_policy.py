import unittest

from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.realtime.policy import PolicyAction, PolicyDecision
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


if __name__ == "__main__":
    unittest.main()
