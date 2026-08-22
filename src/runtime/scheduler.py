"""Generic priority-based asynchronous runtime scheduler."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .priority import Priority


RuntimeTask = Callable[[], Awaitable[Any]] | Awaitable[Any]
_QueueItem = tuple[float, int, RuntimeTask | None]


class RealtimeScheduler:
    """Execute submitted asynchronous tasks in priority order.

    Tasks are started by a single scheduler worker, so the scheduler itself
    never blocks the event loop while a task is awaiting. Tasks with equal
    priority retain submission order. Shutdown is graceful: queued tasks are
    drained before the scheduler stops accepting work.
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._sequence = 0
        self._worker_task: asyncio.Task[None] | None = None
        self._accepting = True
        self._shutdown_requested = False
        self._errors: list[BaseException] = []

    async def submit(self, task: RuntimeTask, priority: Priority) -> None:
        """Queue one async task for execution at ``priority``.

        ``task`` may be an async callable or an already-created awaitable.
        Submission does not wait for task execution.
        """
        if not self._accepting:
            raise RuntimeError("scheduler has been shut down")
        if not isinstance(priority, Priority):
            raise TypeError("priority must be a Priority value")
        if not callable(task) and not inspect.isawaitable(task):
            raise TypeError("task must be an async callable or awaitable")

        self._queue.put_nowait((-priority.value, self._sequence, task))
        self._sequence += 1
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        """Stop accepting tasks and wait for all queued work to finish."""
        if self._shutdown_requested:
            if self._worker_task is not None:
                await self._worker_task
            return

        self._shutdown_requested = True
        self._accepting = False
        if self._worker_task is None:
            return

        self._queue.put_nowait((float("inf"), self._sequence, None))
        self._sequence += 1
        await self._worker_task

    async def _run(self) -> None:
        while True:
            _, _, task = await self._queue.get()
            try:
                if task is None:
                    return
                result = task() if callable(task) else task
                await result
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._errors.append(error)
            finally:
                self._queue.task_done()


Scheduler = RealtimeScheduler
