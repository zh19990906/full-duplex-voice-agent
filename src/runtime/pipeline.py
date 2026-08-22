"""Minimal worker collection and lifecycle orchestration."""

from collections.abc import Iterable

from .worker import Worker


class Pipeline:
    """Manage registered workers without implementing application flow."""

    def __init__(self, workers: Iterable[Worker] = ()) -> None:
        self._workers: list[Worker] = list(workers)

    @property
    def workers(self) -> tuple[Worker, ...]:
        """Return workers in their registration order."""
        return tuple(self._workers)

    def register_worker(self, worker: Worker) -> None:
        """Register one worker for future lifecycle operations."""
        self._workers.append(worker)

    async def start(self) -> None:
        """Start workers in registration order."""
        for worker in self._workers:
            await worker.start()

    async def stop(self) -> None:
        """Stop workers in reverse registration order."""
        for worker in reversed(self._workers):
            await worker.stop()
