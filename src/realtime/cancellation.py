"""Cancellation primitives for queued and already-running realtime work."""

import asyncio
from collections.abc import Awaitable
import inspect


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
        self._task: asyncio.Future[object] | None = None
        self._lock = asyncio.Lock()
        self._last_error: BaseException | None = None

    @property
    def task(self) -> asyncio.Future[object] | None:
        """Return the task currently owned by this slot."""
        return self._task

    @property
    def last_error(self) -> BaseException | None:
        """Return the most recent non-cancellation task failure."""
        return self._last_error

    async def replace(self, work: Awaitable[object]) -> asyncio.Future[object]:
        """Cancel and await current work before scheduling its replacement."""
        installed = False
        try:
            async with self._lock:
                await self._clear_current_task()
                replacement = asyncio.ensure_future(work)
                self._task = replacement
                installed = True
                # Let a newly installed coroutine enter its cancellation cleanup
                # region before a later replacement can preempt it.
                await asyncio.sleep(0)
                return replacement
        except BaseException:
            if not installed:
                self._discard(work)
            raise

    async def cancel(self) -> None:
        """Cancel and await the owned task, if any."""
        async with self._lock:
            await self._clear_current_task()

    async def _clear_current_task(self) -> None:
        task = self._task
        if task is None:
            return

        try:
            task.cancel()
            await self._record_task_outcome(task)
        finally:
            if self._task is task:
                self._task = None

    async def _record_task_outcome(self, task: asyncio.Future[object]) -> None:
        caller_cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    break
                caller_cancelled = True
            except BaseException:
                break

        if task.cancelled():
            if caller_cancelled:
                raise asyncio.CancelledError
            return

        try:
            task.result()
        except BaseException as error:
            self._last_error = error

        if caller_cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _discard(work: Awaitable[object]) -> None:
        """Close a supplied coroutine if replacement cannot install it."""
        if inspect.iscoroutine(work):
            work.close()
