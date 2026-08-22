import asyncio
import unittest

from src.core.events.events import UserBackchannelEvent, UserInterruptEvent
from src.demo.controller import ConversationController
from src.demo.mock_llm import MockLLMAdapter
from src.demo.mock_tts import MockTTSAdapter


class DemoPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await asyncio.sleep(0)

    async def test_backchannel_keeps_generation_active(self):
        llm = MockLLMAdapter(
            response="北京旅游有三个主要景点",
            token_delay=0.02,
        )
        tts = MockTTSAdapter(chunk_delay=0.02)
        controller = ConversationController(llm, tts)

        await controller.start_response("北京旅游")
        await asyncio.sleep(0.01)
        await controller.handle_event(
            UserBackchannelEvent("event-1", 1.0, "mock-turn", {"text": "嗯嗯"})
        )

        self.assertTrue(llm.is_generating)
        self.assertFalse(llm.cancelled)
        self.assertEqual(controller.backchannel_count, 1)

        await controller.stop_response()

    async def test_interrupt_cancels_generation(self):
        llm = MockLLMAdapter(
            response="北京旅游有三个主要景点",
            token_delay=0.02,
        )
        tts = MockTTSAdapter(chunk_delay=0.02)
        controller = ConversationController(llm, tts)

        await controller.start_response("北京旅游")
        await asyncio.sleep(0.01)
        event = UserInterruptEvent(
            "event-2", 2.0, "mock-turn", {"text": "等等，我想问上海"}
        )

        await controller.handle_event(event)

        self.assertTrue(llm.cancelled)
        self.assertFalse(llm.is_generating)
        self.assertEqual(controller.last_interrupt, event)

    async def test_interrupt_triggers_tts_interrupt(self):
        llm = MockLLMAdapter(response="北京旅游有三个主要景点", token_delay=0.02)
        tts = MockTTSAdapter(chunk_delay=0.02)
        controller = ConversationController(llm, tts)

        await controller.start_response("北京旅游")
        await asyncio.sleep(0.01)
        await controller.handle_event(
            UserInterruptEvent("event-3", 3.0, "mock-turn", {"text": "等等"})
        )

        self.assertTrue(tts.interrupted)
        self.assertEqual(tts.interrupt_count, 1)
