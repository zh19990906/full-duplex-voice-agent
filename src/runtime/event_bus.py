"""Async event transport without routing or business logic."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from src.core.events.events import BaseEvent


EventHandler = Callable[[BaseEvent], Awaitable[None]]


class EventBus:
    """Publish events to all handlers subscribed to their type or name."""

    def __init__(self) -> None:
        self._subscribers: dict[Any, list[EventHandler]] = defaultdict(list)

    async def subscribe(self, event_type: Any, handler: EventHandler) -> None:
        """Register an asynchronous handler for an event type or protocol name."""
        self._subscribers[event_type].append(handler)

    async def unsubscribe(self, event_type: Any, handler: EventHandler) -> None:
        """Remove a previously registered handler when present."""
        handlers = self._subscribers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            del self._subscribers[event_type]

    async def publish(self, event: BaseEvent) -> None:
        """Deliver an event to type and protocol-name subscribers."""
        handlers = [
            *self._subscribers.get(type(event), []),
            *self._subscribers.get(event.event, []),
        ]
        await asyncio.gather(*(handler(event) for handler in handlers))
