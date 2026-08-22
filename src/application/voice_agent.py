"""MVP voice-agent event orchestration."""

from src.controller.controller import ConversationController
from src.controller.actions import ControllerAction
from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus

from .actions_executor import ActionExecutor


class VoiceAgent:
    """Connect the event bus, pure controller, and action executor."""

    _EVENT_TYPES = (
        UserBackchannelEvent,
        UserInterruptEvent,
        UserTurnEndEvent,
    )

    def __init__(
        self,
        event_bus: EventBus,
        controller: ConversationController,
        generation_manager: GenerationManager,
    ) -> None:
        self.event_bus = event_bus
        self.controller = controller
        self.generation_manager = generation_manager
        self.action_executor = ActionExecutor(generation_manager)
        self._started = False

    async def start(self) -> None:
        """Subscribe the application handler to user control events."""
        if self._started:
            return
        for event_type in self._EVENT_TYPES:
            await self.event_bus.subscribe(event_type, self.handle_event)
        self._started = True

    async def stop(self) -> None:
        """Remove application subscriptions from the event bus."""
        if not self._started:
            return
        for event_type in self._EVENT_TYPES:
            await self.event_bus.unsubscribe(event_type, self.handle_event)
        self._started = False

    async def handle_event(self, event: BaseEvent) -> tuple[ControllerAction, ...]:
        """Transform one event through the controller and execute its actions."""
        actions = self.controller.handle_event(event)
        await self.action_executor.execute(actions)
        return actions
