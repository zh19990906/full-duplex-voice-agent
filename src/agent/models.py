"""Serializable state and decision models for the agent loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    OBSERVING = "OBSERVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class AgentState:
    """Current lifecycle snapshot for one bounded agent run."""

    session_id: str
    iteration: int = 0
    status: AgentStatus = AgentStatus.IDLE
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("agent session_id must not be empty")
        if self.iteration < 0:
            raise ValueError("agent iteration must not be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True)
class ToolCallRequest:
    """Provider-neutral tool decision emitted by an LLM-facing callback."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tool call requires a non-empty name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")
        object.__setattr__(self, "arguments", dict(self.arguments))
        if not self.call_id:
            object.__setattr__(self, "call_id", uuid4().hex)

    @property
    def tool_name(self) -> str:
        """Stable semantic alias for the legacy ``name`` field."""

        return self.name

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolCallRequest":
        name = value.get("tool", value.get("tool_name", value.get("name")))
        arguments = value.get("arguments", {})
        call_id = value.get("call_id", value.get("id", ""))
        if not isinstance(name, str) or not name:
            raise ValueError("tool call requires a non-empty tool name")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool call arguments must be an object")
        return cls(name=name, arguments=dict(arguments), call_id=str(call_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.name,
            "tool_name": self.name,
            "arguments": dict(self.arguments),
        }
