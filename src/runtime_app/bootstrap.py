"""Application startup entry points without model loading or servers."""

from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler

from .container import ApplicationContainer


def create_application(
    *,
    event_bus: EventBus | None = None,
    controller: ConversationController | None = None,
    generation_manager: GenerationManager | None = None,
    scheduler: RealtimeScheduler | None = None,
) -> ApplicationContainer:
    """Create an uninitialized application container."""
    return ApplicationContainer(
        event_bus=event_bus,
        controller=controller,
        generation_manager=generation_manager,
        scheduler=scheduler,
    )


async def initialize_application(
    *,
    event_bus: EventBus | None = None,
    controller: ConversationController | None = None,
    generation_manager: GenerationManager | None = None,
    scheduler: RealtimeScheduler | None = None,
) -> ApplicationContainer:
    """Create and initialize an application container."""
    application = create_application(
        event_bus=event_bus,
        controller=controller,
        generation_manager=generation_manager,
        scheduler=scheduler,
    )
    await application.initialize()
    return application
