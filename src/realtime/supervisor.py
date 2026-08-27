"""Minimal worker supervision with a single terminal event contract."""

from __future__ import annotations

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

    def __init__(self, worker: str, *, max_restarts: int = 1) -> None:
        if not isinstance(worker, str) or not worker.strip():
            raise ValueError("worker must be a non-empty string")
        if type(max_restarts) is not int or max_restarts < 0:
            raise ValueError("max_restarts must be a non-negative integer")
        self.worker = worker
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.status = "IDLE"
        self.last_error: BaseException | None = None

    async def run(
        self,
        operation: Any,
        *,
        on_restart: Any | None = None,
    ) -> Any:
        self.status = "RUNNING"
        while True:
            try:
                result = operation() if callable(operation) else operation
                if inspect.isawaitable(result):
                    result = await result
                self.status = "COMPLETED"
                return result
            except BaseException as exc:
                self.last_error = exc
                if self.restart_count >= self.max_restarts:
                    self.status = "FAILED"
                    return WorkerTerminalEvent(
                        worker=self.worker,
                        message=str(exc),
                        restart_count=self.restart_count,
                        error_type=type(exc).__name__,
                    )
                self.restart_count += 1
                if on_restart is not None:
                    restart_result = on_restart(exc, self.restart_count)
                    if inspect.isawaitable(restart_result):
                        await restart_result

    async def recover(
        self,
        error: BaseException,
        *,
        on_restart: Any | None = None,
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
        self.status = "RUNNING"
        self.restart_count += 1
        if on_restart is not None:
            result = on_restart(error, self.restart_count)
            if inspect.isawaitable(result):
                await result
        return True
