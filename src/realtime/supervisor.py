"""Minimal worker supervision with a single terminal event contract."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerTerminalEvent:
    """Serializable terminal worker failure for browser/event-bus delivery."""

    worker: str
    message: str
    restart_count: int
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "worker_terminal",
            "payload": {
                "worker": self.worker,
                "message": self.message,
                "restart_count": self.restart_count,
                "error_type": self.error_type,
            },
        }


class WorkerSupervisor:
    """Retry one failing worker operation and surface terminal failure explicitly."""

    def __init__(
        self,
        worker: str,
        *,
        max_restarts: int = 1,
        restart_backoff_seconds: float = 0.1,
        sleep: Any = asyncio.sleep,
    ) -> None:
        if not isinstance(worker, str) or not worker.strip():
            raise ValueError("worker must be a non-empty string")
        if type(max_restarts) is not int or max_restarts < 0:
            raise ValueError("max_restarts must be a non-negative integer")
        if (
            isinstance(restart_backoff_seconds, bool)
            or not isinstance(restart_backoff_seconds, (int, float))
            or restart_backoff_seconds < 0
        ):
            raise ValueError("restart_backoff_seconds must be non-negative")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self.worker = worker
        self.max_restarts = max_restarts
        self.restart_backoff_seconds = float(restart_backoff_seconds)
        self._sleep = sleep
        self.restart_count = 0
        self.status = "IDLE"
        self.last_error: BaseException | None = None
        self._active_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def run(
        self,
        operation: Any,
        *,
        on_restart: Any | None = None,
    ) -> Any:
        self._ensure_open()
        current_task = asyncio.current_task()
        self._active_task = current_task
        self.status = "RUNNING"
        try:
            while True:
                try:
                    result = operation() if callable(operation) else operation
                    if inspect.isawaitable(result):
                        result = await result
                    self.status = "COMPLETED"
                    return result
                except asyncio.CancelledError:
                    self.status = "CLOSED" if self._closed else "CANCELLED"
                    raise
                except Exception as exc:
                    terminal = await self._restart_or_terminal(exc, on_restart=on_restart)
                    if terminal is not True:
                        return terminal
        except asyncio.CancelledError:
            self.status = "CLOSED" if self._closed else "CANCELLED"
            raise
        finally:
            if self._active_task is current_task:
                self._active_task = None

    async def recover(
        self,
        error: BaseException,
        *,
        on_restart: Any | None = None,
    ) -> bool | WorkerTerminalEvent:
        self._ensure_open()
        if isinstance(error, asyncio.CancelledError):
            self.status = "CANCELLED"
            raise error
        current_task = asyncio.current_task()
        self._active_task = current_task
        try:
            return await self._restart_or_terminal(error, on_restart=on_restart)
        except asyncio.CancelledError:
            self.status = "CLOSED" if self._closed else "CANCELLED"
            raise
        finally:
            if self._active_task is current_task:
                self._active_task = None

    async def close(self) -> None:
        """Cancel pending supervision work exactly once."""
        task: asyncio.Task[Any] | None = None
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.status = "CLOSED"
            active = self._active_task
            if active is not None and active is not asyncio.current_task() and not active.done():
                task = active
                task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _restart_or_terminal(
        self,
        error: BaseException,
        *,
        on_restart: Any | None,
    ) -> bool | WorkerTerminalEvent:
        self.last_error = error
        if self.restart_count >= self.max_restarts:
            self.status = "FAILED"
            return WorkerTerminalEvent(
                worker=self.worker,
                message=str(error),
                restart_count=self.restart_count,
                error_type=type(error).__name__,
            )
        self.status = "BACKING_OFF"
        self.restart_count += 1
        await self._sleep(self.restart_backoff_seconds)
        self._ensure_open()
        self.status = "RUNNING"
        if on_restart is not None:
            result = on_restart(error, self.restart_count)
            if inspect.isawaitable(result):
                await result
        return True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.worker} supervisor is closed")
