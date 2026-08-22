"""Model-independent asynchronous worker abstraction."""

from src.core.events.events import BaseEvent


class Worker:
    """Base worker with overridable lifecycle and event hooks."""

    async def start(self) -> None:
        """Start the worker."""
        return None

    async def stop(self) -> None:
        """Stop the worker."""
        return None

    async def handle_event(self, event: BaseEvent) -> None:
        """Handle one event when a runtime dispatches it."""
        return None
