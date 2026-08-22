"""Dependency container for the application runtime composition."""

from src.application.voice_agent import VoiceAgent
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler


class ApplicationContainer:
    """Create and wire the model-independent application components.

    Dependencies are accepted as optional constructor arguments so tests and
    future deployments can replace infrastructure without changing the
    composition boundary. Model adapters are intentionally not constructed.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        controller: ConversationController | None = None,
        generation_manager: GenerationManager | None = None,
        scheduler: RealtimeScheduler | None = None,
    ) -> None:
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.controller = controller if controller is not None else ConversationController()
        self.generation_manager = (
            generation_manager if generation_manager is not None else GenerationManager()
        )
        self.scheduler = scheduler if scheduler is not None else RealtimeScheduler()
        self.voice_agent = VoiceAgent(
            event_bus=self.event_bus,
            controller=self.controller,
            generation_manager=self.generation_manager,
        )
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Whether application event subscriptions are active."""
        return self._initialized

    async def initialize(self) -> None:
        """Start application-level event subscriptions."""
        if self._initialized:
            return
        await self.voice_agent.start()
        self._initialized = True

    async def shutdown(self) -> None:
        """Stop application subscriptions and the scheduler."""
        await self.voice_agent.stop()
        await self.scheduler.shutdown()
        self._initialized = False
