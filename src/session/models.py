"""Serializable session and resource ownership models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class SessionResource:
    """A logical resource owned by one session."""

    session_id: str
    resource_type: str
    owner: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class Session:
    """Session identity, lifecycle state, and session-owned resources."""

    session_id: str
    created_at: float = field(default_factory=time)
    status: SessionStatus = SessionStatus.CREATED
    metadata: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, SessionResource] = field(default_factory=dict, repr=False)
    memory_manager: Any = field(default=None, repr=False, compare=False)
    agent: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        self.created_at = float(self.created_at)
        self.metadata = dict(self.metadata)
        self.resources = dict(self.resources)

    def activate(self) -> None:
        if self.status not in {SessionStatus.CREATED, SessionStatus.IDLE}:
            raise ValueError(f"cannot activate session in {self.status.value} state")
        self.status = SessionStatus.ACTIVE

    def mark_idle(self) -> None:
        if self.status is not SessionStatus.ACTIVE:
            raise ValueError(f"cannot mark session idle from {self.status.value} state")
        self.status = SessionStatus.IDLE

    def begin_closing(self) -> None:
        if self.status not in {SessionStatus.CREATED, SessionStatus.ACTIVE, SessionStatus.IDLE}:
            raise ValueError(f"cannot close session in {self.status.value} state")
        self.status = SessionStatus.CLOSING

    def close(self) -> None:
        if self.status is not SessionStatus.CLOSING:
            raise ValueError(f"cannot complete close from {self.status.value} state")
        self.status = SessionStatus.CLOSED

    def fail(self) -> None:
        if self.status is SessionStatus.CLOSED:
            raise ValueError("cannot fail a closed session")
        self.status = SessionStatus.FAILED

    def add_resource(self, resource_type: str) -> SessionResource:
        if not resource_type:
            raise ValueError("resource_type must not be empty")
        resource = SessionResource(self.session_id, resource_type, self.session_id)
        self.resources[resource_type] = resource
        return resource

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "resources": {name: resource.to_dict() for name, resource in self.resources.items()},
        }
