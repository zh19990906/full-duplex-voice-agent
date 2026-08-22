"""Serializable tool request, definition, and result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping


ToolHandler = Callable[..., Any]


@dataclass
class ToolDefinition:
    """A tool contract plus its application-owned execution callback."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        self.parameters = dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Return discovery metadata without exposing the callback object."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ToolRequest:
    """Normalized request from an LLM/application decision."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, request: Mapping[str, Any]) -> "ToolRequest":
        name = request.get("name")
        arguments = request.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("tool request requires a non-empty name")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool request arguments must be an object")
        return cls(name=name, arguments=dict(arguments))


@dataclass(frozen=True)
class ToolResult:
    """Isolated success/failure result returned by tool execution."""

    success: bool
    output: Any = None
    error: str | None = None
    tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
