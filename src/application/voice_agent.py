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
from src.memory.context import ContextBuilder
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.tools.models import ToolRequest, ToolResult
from src.tools.router import ToolRouter

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
        memory_manager: MemoryManager | None = None,
        memory_session_id: str = "default",
        context_builder: ContextBuilder | None = None,
        tool_router: ToolRouter | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.controller = controller
        self.generation_manager = generation_manager
        self.action_executor = ActionExecutor(generation_manager)
        self.memory_manager = memory_manager
        self.memory_session_id = memory_session_id
        self.context_builder = context_builder or (
            ContextBuilder(memory_manager) if memory_manager is not None else None
        )
        self.generation_context = ()
        self.tool_router = tool_router
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
        if isinstance(event, UserTurnEndEvent) and self.memory_manager is not None:
            text = event.payload.get("text")
            if isinstance(text, str) and text:
                self._ensure_memory_session()
                self.memory_manager.add_message(
                    self.memory_session_id,
                    "user",
                    text,
                    timestamp=event.timestamp,
                    metadata={"session_id": self.memory_session_id, "source": event.source},
                )
                self.generation_context = self.context_builder.build(self.memory_session_id)
        return actions

    async def execute_tool(self, request: ToolRequest | dict[str, object]) -> ToolResult:
        """Execute one application tool decision through the configured router."""

        if self.tool_router is None:
            raise RuntimeError("tool calling is not configured")
        return await self.tool_router.route(request)

    def record_assistant_response(
        self,
        content: str,
        timestamp: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Store a completed assistant response for the next generation."""

        if self.memory_manager is None:
            raise RuntimeError("conversation memory is not configured")
        self._ensure_memory_session()
        self.memory_manager.add_message(
            self.memory_session_id,
            "assistant",
            content,
            timestamp=timestamp,
            metadata={"session_id": self.memory_session_id, **(metadata or {})},
        )
        self.generation_context = self.context_builder.build(self.memory_session_id)

    def _ensure_memory_session(self) -> None:
        try:
            self.memory_manager.get_session(self.memory_session_id)
        except KeyError:
            self.memory_manager.create_session(self.memory_session_id)
