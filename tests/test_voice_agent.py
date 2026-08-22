import unittest

from src.application.voice_agent import VoiceAgent
from src.controller.actions import ActionType
from src.controller.controller import ConversationController
from src.controller.states import ControllerState
from src.core.events.events import (
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus


def make_event(event_type, event_id="event-1"):
    return event_type(event_id, 1.0, "test", {})


class VoiceAgentTest(unittest.IsolatedAsyncioTestCase):
    async def create_agent(self, state=ControllerState.SPEAKING):
        event_bus = EventBus()
        controller = ConversationController(state=state)
        generation_manager = GenerationManager()
        agent = VoiceAgent(event_bus, controller, generation_manager)
        await agent.start()
        return agent, event_bus, controller, generation_manager

    async def test_backchannel_keeps_active_generation_and_records_continue(self):
        agent, event_bus, controller, generation_manager = await self.create_agent()
        session = await generation_manager.start_generation("generation-1")

        await event_bus.publish(make_event(UserBackchannelEvent))

        self.assertIs(generation_manager.active_session, session)
        self.assertEqual(controller.state, ControllerState.SPEAKING)
        self.assertEqual(
            [action.action_type for action in agent.action_executor.history],
            [ActionType.CONTINUE_GENERATION],
        )
        await agent.stop()

    async def test_interrupt_cancels_active_generation_through_executor(self):
        agent, event_bus, controller, generation_manager = await self.create_agent()
        session = await generation_manager.start_generation("generation-2")

        await event_bus.publish(make_event(UserInterruptEvent, "event-2"))

        self.assertEqual(controller.state, ControllerState.INTERRUPTED)
        self.assertEqual(session.status.value, "CANCELLED")
        self.assertIsNone(generation_manager.active_session)
        self.assertEqual(
            [action.action_type for action in agent.action_executor.history],
            [ActionType.STOP_RESPONSE, ActionType.CANCEL_GENERATION],
        )
        await agent.stop()

    async def test_turn_end_updates_controller_without_model_invocation(self):
        agent, event_bus, controller, generation_manager = await self.create_agent(
            state=ControllerState.LISTENING
        )

        await event_bus.publish(make_event(UserTurnEndEvent, "event-3"))

        self.assertEqual(controller.state, ControllerState.THINKING)
        self.assertEqual(
            [action.action_type for action in agent.action_executor.history],
            [ActionType.PROCESS_USER_REQUEST],
        )
        self.assertIsNone(generation_manager.active_session)
        await agent.stop()
