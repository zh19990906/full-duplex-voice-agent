import unittest

from src.controller.actions import ActionType, ControllerAction
from src.controller.controller import ConversationController
from src.controller.states import ControllerState
from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)


def event(event_type):
    return event_type("event-1", 1.0, "test", {})


class ConversationControllerTest(unittest.TestCase):
    def test_initial_state_is_idle(self):
        controller = ConversationController()

        self.assertEqual(controller.state, ControllerState.IDLE)

    def test_backchannel_while_speaking_keeps_state_and_continues_generation(self):
        controller = ConversationController(state=ControllerState.SPEAKING)

        actions = controller.handle_event(event(UserBackchannelEvent))

        self.assertEqual(controller.state, ControllerState.SPEAKING)
        self.assertEqual(
            actions,
            (ControllerAction(ActionType.CONTINUE_GENERATION),),
        )

    def test_interrupt_transitions_to_interrupted_and_returns_stop_cancel(self):
        controller = ConversationController(state=ControllerState.SPEAKING)

        actions = controller.handle_event(event(UserInterruptEvent))

        self.assertEqual(controller.state, ControllerState.INTERRUPTED)
        self.assertEqual(
            actions,
            (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
            ),
        )

    def test_turn_end_transitions_to_thinking_and_processes_request(self):
        controller = ConversationController(state=ControllerState.LISTENING)

        actions = controller.handle_event(event(UserTurnEndEvent))

        self.assertEqual(controller.state, ControllerState.THINKING)
        self.assertEqual(
            actions,
            (ControllerAction(ActionType.PROCESS_USER_REQUEST),),
        )

    def test_unknown_event_returns_no_actions_without_changing_state(self):
        controller = ConversationController(state=ControllerState.LISTENING)

        actions = controller.handle_event(event(BaseEvent))

        self.assertEqual(controller.state, ControllerState.LISTENING)
        self.assertEqual(actions, ())
