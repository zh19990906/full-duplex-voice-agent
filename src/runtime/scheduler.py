"""Generic priority-based asynchronous runtime scheduler."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.realtime.cancellation import CancellationToken

from .priority import Priority


RuntimeTask = Callable[[], Awaitable[Any]] | Awaitable[Any]


@dataclass(frozen=True)
class _ScheduledTask:
    task: RuntimeTask
    deadline: float
    cancellation_token: CancellationToken


_QueueItem = tuple[float, int, _ScheduledTask | None]


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

    async def submit(
        self,
        task: RuntimeTask,
        priority: Priority,
        *,
        deadline: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
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

        if deadline is None:
            deadline = float("inf")
        if cancellation_token is None:
            cancellation_token = CancellationToken()
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken value")

        scheduled_task = _ScheduledTask(task, deadline, cancellation_token)
        self._queue.put_nowait((-priority.value, self._sequence, scheduled_task))
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
            _, _, scheduled_task = await self._queue.get()
            try:
                if scheduled_task is None:
                    return
                if (
                    scheduled_task.cancellation_token.is_cancelled()
                    or asyncio.get_running_loop().time() >= scheduled_task.deadline
                ):
                    self._discard(scheduled_task.task)
                    continue
                task = scheduled_task.task
                result = task() if callable(task) else task
                await result
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._errors.append(error)
            finally:
                self._queue.task_done()

    @staticmethod
    def _discard(task: RuntimeTask) -> None:
        """Close unstarted coroutine objects skipped while queued."""
        if inspect.iscoroutine(task):
            task.close()


Scheduler = RealtimeScheduler
