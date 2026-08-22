"""Executable scenarios for the first vertical slice."""

import asyncio

from src.core.events.events import UserBackchannelEvent, UserInterruptEvent
from src.runtime.event_bus import EventBus

from .controller import ConversationController
from .mock_llm import MockLLMAdapter
from .mock_tts import MockTTSAdapter
from .mock_turn import MockTurnAdapter


async def run_demo() -> None:
    """Run backchannel-continuation and interruption scenarios."""
    bus = EventBus()
    turn = MockTurnAdapter()
    llm = MockLLMAdapter(token_delay=0.02)
    tts = MockTTSAdapter(chunk_delay=0.02)
    controller = ConversationController(llm, tts)

    await bus.subscribe(UserBackchannelEvent, controller.handle_event)
    await bus.subscribe(UserInterruptEvent, controller.handle_event)

    print("Scenario 1: backchannel keeps the response active")
    await controller.start_response("北京旅游")
    await asyncio.sleep(0.03)
    await bus.publish(turn.backchannel())
    print(f"generation active: {llm.is_generating}")
    await controller.stop_response()

    print("Scenario 2: interruption stops speech and generation")
    await controller.start_response("北京旅游")
    await asyncio.sleep(0.03)
    await bus.publish(turn.interrupt())
    print(f"generation cancelled: {llm.cancelled}")
    print(f"tts interrupted: {tts.interrupted}")


if __name__ == "__main__":
    asyncio.run(run_demo())
