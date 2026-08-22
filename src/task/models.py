"""Generic task state data model."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle metadata for a task state record."""

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass
class TaskState:
    """Task progress data without task execution behavior."""

    task_id: str
    task_type: str
    status: TaskStatus
    state_data: dict[str, Any]
