"""Generic task state lifecycle manager."""

from typing import Any

from .models import TaskState, TaskStatus
from .storage import InMemoryTaskStorage, TaskStorage


class TaskStateManager:
    """Create and update task metadata without executing tasks."""

    def __init__(self, storage: TaskStorage | None = None) -> None:
        self.storage = storage or InMemoryTaskStorage()

    def create_task(
        self,
        task_id: str,
        task_type: str,
        state_data: dict[str, Any],
    ) -> TaskState:
        """Create a running task state record."""
        if self.storage.load(task_id) is not None:
            raise ValueError(f"Task already exists: {task_id}")
        task = TaskState(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.RUNNING,
            state_data=dict(state_data),
        )
        self.storage.save(task)
        return task

    def update_state(self, task_id: str, state_data: dict[str, Any]) -> TaskState:
        """Replace the generic state data for an existing task."""
        task = self._require_task(task_id)
        task.state_data = dict(state_data)
        self.storage.save(task)
        return task

    def get_task(self, task_id: str) -> TaskState:
        """Return an existing task state or raise KeyError."""
        return self._require_task(task_id)

    def pause_task(self, task_id: str) -> TaskState:
        """Transition a running task to paused metadata state."""
        task = self._require_task(task_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task is not running: {task_id}")
        task.status = TaskStatus.PAUSED
        self.storage.save(task)
        return task

    def resume_task(self, task_id: str) -> TaskState:
        """Transition a paused task back to running metadata state."""
        task = self._require_task(task_id)
        if task.status is not TaskStatus.PAUSED:
            raise ValueError(f"Task is not paused: {task_id}")
        task.status = TaskStatus.RUNNING
        self.storage.save(task)
        return task

    def _require_task(self, task_id: str) -> TaskState:
        task = self.storage.load(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        return task
