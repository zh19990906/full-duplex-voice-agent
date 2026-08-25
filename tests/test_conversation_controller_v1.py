import unittest

from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.session_state import ConversationMode, FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent


def decision(action, **fields):
    return PolicyDecision(
        action=action,
        confidence=0.98,
        rationale="test decision",
        **fields,
    )


def candidate(event, label, confidence=0.9):
    return SpeechCandidateEvent(
        event=event,
        event_id="candidate-1",
        timestamp=1.0,
        source="test",
        payload={"label": label, "confidence": confidence, "evidence": {}},
    )


class ConversationControllerV1Tests(unittest.TestCase):
    def test_controller_action_payload_is_deeply_immutable_snapshot(self):
        original = {"language": {"source": "Chinese"}, "alternates": ["English"]}
        action = ControllerAction(ActionType.SWITCH_MODE, original)

        original["language"]["source"] = "French"
        original["alternates"].append("German")

        self.assertEqual(action.payload["language"]["source"], "Chinese")
        self.assertEqual(action.payload["alternates"], ("English",))
        with self.assertRaises(TypeError):
            action.payload["language"]["source"] = "Spanish"

    def test_action_enum_preserves_legacy_values_and_adds_realtime_actions(self):
        self.assertEqual(ActionType.CONTINUE_GENERATION.value, "CONTINUE_GENERATION")
        self.assertEqual(ActionType.CANCEL_GENERATION.value, "CANCEL_GENERATION")
        self.assertEqual(
            {item.value for item in ActionType},
            {
                "CONTINUE_GENERATION", "STOP_RESPONSE", "CANCEL_GENERATION",
                "PROCESS_USER_REQUEST", "DUCK_RESPONSE", "RESTORE_RESPONSE",
                "PAUSE_RESPONSE", "RESUME_RESPONSE", "REVISE_RESPONSE",
                "SWITCH_MODE", "REQUEST_CLARIFICATION",
            },
        )

    def test_candidate_backchannel_ducks_playback_without_committing_semantics(self):
        state = SessionState(ConversationMode.CHAT, FloorState.ASSISTANT, ResponseState.PLAYING)
        actions = ConversationController().handle_candidate(
            state, candidate("USER_BACKCHANNEL_CANDIDATE", "backchannel")
        )
        self.assertEqual([item.action_type for item in actions], [ActionType.DUCK_RESPONSE])
        self.assertEqual((state.floor, state.response), (FloorState.OVERLAP, ResponseState.DUCKED))

    def test_candidate_interrupt_pauses_without_cancelling_or_processing(self):
        state = SessionState(ConversationMode.CHAT, FloorState.ASSISTANT, ResponseState.PLAYING)
        actions = ConversationController().handle_candidate(
            state, candidate("USER_SPEECH_START_CANDIDATE", "speaking")
        )
        self.assertEqual([item.action_type for item in actions], [ActionType.PAUSE_RESPONSE])
        self.assertEqual((state.floor, state.response), (FloorState.OVERLAP, ResponseState.PAUSED))

    def test_affirmation_after_question_is_answer_not_backchannel(self):
        controller = ConversationController()
        state = SessionState(
            mode=ConversationMode.CHAT,
            floor=FloorState.OVERLAP,
            response=ResponseState.DUCKED,
            assistant_act="ASKING",
        )
        policy = PolicyDecision(
            action=PolicyAction.ANSWER,
            confidence=0.98,
            rationale="The user answered the assistant's question.",
        )
        actions = controller.apply_policy(state, policy)
        assert [item.action_type for item in actions] == [ActionType.STOP_RESPONSE, ActionType.PROCESS_USER_REQUEST]
        self.assertEqual((state.floor, state.response), (FloorState.USER, ResponseState.IDLE))

    def test_backchannel_restores_ducked_response_and_continues(self):
        state = SessionState(ConversationMode.CHAT, FloorState.OVERLAP, ResponseState.DUCKED)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.BACKCHANNEL))
        self.assertEqual(
            [item.action_type for item in actions],
            [ActionType.RESTORE_RESPONSE, ActionType.CONTINUE_GENERATION],
        )
        self.assertEqual((state.floor, state.response), (FloorState.ASSISTANT, ResponseState.PLAYING))

    def test_pause_preserves_response_without_new_request(self):
        state = SessionState(ConversationMode.CHAT, FloorState.OVERLAP, ResponseState.PLAYING)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.PAUSE))
        self.assertEqual([item.action_type for item in actions], [ActionType.PAUSE_RESPONSE])
        self.assertEqual((state.floor, state.response), (FloorState.USER, ResponseState.PAUSED))

    def test_resume_only_resumes_preserved_response(self):
        state = SessionState(ConversationMode.CHAT, FloorState.USER, ResponseState.PAUSED)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.RESUME))
        self.assertEqual([item.action_type for item in actions], [ActionType.RESUME_RESPONSE])
        self.assertEqual((state.floor, state.response), (FloorState.ASSISTANT, ResponseState.PLAYING))

    def test_revise_stops_cancels_revises_then_processes(self):
        state = SessionState(ConversationMode.CHAT, FloorState.OVERLAP, ResponseState.DUCKED)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.REVISE))
        self.assertEqual(
            [item.action_type for item in actions],
            [
                ActionType.STOP_RESPONSE,
                ActionType.CANCEL_GENERATION,
                ActionType.REVISE_RESPONSE,
                ActionType.PROCESS_USER_REQUEST,
            ],
        )
        self.assertEqual((state.floor, state.response), (FloorState.USER, ResponseState.CANCELLING))

    def test_new_request_stops_cancels_and_processes(self):
        state = SessionState(ConversationMode.CHAT, FloorState.OVERLAP, ResponseState.PLAYING)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.NEW_REQUEST))
        self.assertEqual(
            [item.action_type for item in actions],
            [ActionType.STOP_RESPONSE, ActionType.CANCEL_GENERATION, ActionType.PROCESS_USER_REQUEST],
        )
        self.assertEqual((state.floor, state.response), (FloorState.USER, ResponseState.CANCELLING))

    def test_explicit_policy_mode_switch_sets_language_direction(self):
        state = SessionState(ConversationMode.CHAT, FloorState.USER, ResponseState.IDLE)
        actions = ConversationController().apply_policy(
            state,
            decision(
                PolicyAction.MODE_SWITCH,
                intent="continuous_interpretation",
                source_language="Chinese",
                target_language="English",
            ),
        )
        self.assertEqual([item.action_type for item in actions], [ActionType.SWITCH_MODE])
        self.assertEqual(state.mode, ConversationMode.INTERPRETATION)
        self.assertEqual(
            dict(actions[0].payload),
            {
                "target_mode": "INTERPRETATION",
                "source_language": "Chinese",
                "target_language": "English",
            },
        )

    def test_continuous_interpretation_language_switch_does_not_exit_mode(self):
        state = SessionState(ConversationMode.INTERPRETATION, FloorState.USER, ResponseState.IDLE)
        actions = ConversationController().apply_policy(
            state,
            decision(
                PolicyAction.MODE_SWITCH,
                intent="continuous_interpretation",
                source_language="English",
                target_language="Chinese",
            ),
        )
        self.assertEqual(state.mode, ConversationMode.INTERPRETATION)
        self.assertEqual(dict(actions[0].payload)["target_mode"], "INTERPRETATION")

    def test_explicit_chat_mode_switch_exits_interpretation(self):
        state = SessionState(ConversationMode.INTERPRETATION, FloorState.USER, ResponseState.IDLE)
        actions = ConversationController().apply_policy(
            state,
            decision(PolicyAction.MODE_SWITCH, intent="chat"),
        )
        self.assertEqual(state.mode, ConversationMode.CHAT)
        self.assertEqual(dict(actions[0].payload)["target_mode"], "CHAT")

    def test_uncertain_pauses_and_requests_clarification_never_continues(self):
        state = SessionState(ConversationMode.CHAT, FloorState.OVERLAP, ResponseState.DUCKED)
        actions = ConversationController().apply_policy(state, decision(PolicyAction.UNCERTAIN))
        self.assertEqual(
            [item.action_type for item in actions],
            [ActionType.PAUSE_RESPONSE, ActionType.REQUEST_CLARIFICATION],
        )
        self.assertNotIn(ActionType.CONTINUE_GENERATION, [item.action_type for item in actions])
        self.assertEqual((state.floor, state.response), (FloorState.USER, ResponseState.PAUSED))


if __name__ == "__main__":
    unittest.main()
