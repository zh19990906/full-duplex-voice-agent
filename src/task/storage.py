"""Storage abstraction and in-memory implementation for task state."""

from abc import ABC, abstractmethod
from copy import deepcopy

from .models import TaskState


class TaskStorage(ABC):
    """Minimal storage contract for task state records."""

    @abstractmethod
    def save(self, task: TaskState) -> None:
        """Store a task state record."""
        raise NotImplementedError

    @abstractmethod
    def load(self, task_id: str) -> TaskState | None:
        """Load a task state record when it exists."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, task_id: str) -> None:
        """Delete a task state record when present."""
        raise NotImplementedError


class InMemoryTaskStorage(TaskStorage):
    """Process-local task state storage."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}

    def save(self, task: TaskState) -> None:
        self._tasks[task.task_id] = deepcopy(task)

    def load(self, task_id: str) -> TaskState | None:
        task = self._tasks.get(task_id)
        return deepcopy(task) if task is not None else None

    def delete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
