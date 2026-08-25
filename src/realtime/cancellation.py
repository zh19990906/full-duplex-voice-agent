"""Cancellation primitives for queued and already-running realtime work."""

import asyncio
from collections.abc import Awaitable
from contextlib import suppress


class CancellationToken:
    """A cooperative cancellation signal used before scheduled work starts."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Mark this token as cancelled."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled


class ActiveTaskSlot:
    """Own one workflow task and preempt it before replacing it."""

    def __init__(self) -> None:
        self._task: asyncio.Task[object] | None = None

    @property
    def task(self) -> asyncio.Task[object] | None:
        """Return the task currently owned by this slot."""
        return self._task

    async def replace(self, work: Awaitable[object]) -> asyncio.Task[object]:
        """Cancel and await current work before scheduling its replacement."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

        self._task = asyncio.ensure_future(work)
        # Let a newly installed coroutine enter its cancellation cleanup region
        # before a later replacement can preempt it.
        await asyncio.sleep(0)
        return self._task

    async def cancel(self) -> None:
        """Cancel and await the owned task, if any."""
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
