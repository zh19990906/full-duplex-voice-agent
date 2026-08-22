"""Minimal shared state dataclasses."""

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """State shared by the conversation controller and its collaborators."""

    session_id: str
    current_mode: str = "LISTENING"
    active_task: str | None = None
    current_response_id: str | None = None


@dataclass
class TaskState:
    """Serializable task continuation state without lifecycle behavior."""

    task_type: str
    state: dict = field(default_factory=dict)


@dataclass
class RuntimeState:
    """Shared runtime status, without worker or model lifecycle logic."""

    active_workers: list[str] = field(default_factory=list)
    model_status: dict[str, str] = field(default_factory=dict)
    session_status: str = "IDLE"
