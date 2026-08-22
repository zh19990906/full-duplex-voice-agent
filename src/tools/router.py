"""Tool request validation, routing, execution, and memory recording."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import time
from typing import Any

from src.memory.manager import MemoryManager

from .executor import ToolExecutor
from .models import ToolRequest, ToolResult
from .registry import ToolRegistry


class ToolRouter:
    """Route normalized tool decisions to registered application tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor | None = None,
        memory_manager: MemoryManager | None = None,
        session_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.executor = executor or ToolExecutor()
        self.memory_manager = memory_manager
        self.session_id = session_id

    async def route(self, request: ToolRequest | Mapping[str, Any]) -> ToolResult:
        try:
            normalized = request if isinstance(request, ToolRequest) else ToolRequest.from_mapping(request)
            tool = self.registry.get_tool(normalized.name)
            self._validate_arguments(tool.parameters, normalized.arguments)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        self._record_call(normalized)
        result = await self.executor.execute(tool, normalized.arguments)
        self._record_result(result)
        return result

    @staticmethod
    def _validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
        if schema.get("type", "object") != "object":
            raise ValueError("tool parameter schema must describe an object")
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if schema.get("additionalProperties", True) is False:
            unexpected = set(arguments) - set(properties)
            if unexpected:
                raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
        for name in required:
            if name not in arguments:
                raise ValueError(f"missing required argument: {name}")
        for name, value in arguments.items():
            expected = properties.get(name, {}).get("type")
            if expected is not None and not ToolRouter._matches_type(value, expected):
                raise ValueError(f"argument {name!r} must be of type {expected}")

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        return {
            "string": lambda: isinstance(value, str),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": lambda: isinstance(value, bool),
            "array": lambda: isinstance(value, list),
            "object": lambda: isinstance(value, Mapping),
        }.get(expected, lambda: True)()

    def _record_call(self, request: ToolRequest) -> None:
        if self.memory_manager is None or self.session_id is None:
            return
        self._ensure_session()
        self.memory_manager.add_message(
            self.session_id,
            "system",
            json.dumps({"tool": request.name, "arguments": dict(request.arguments)}, sort_keys=True),
            timestamp=time(),
            metadata={"kind": "tool_call", "tool_name": request.name, "session_id": self.session_id},
        )

    def _record_result(self, result: ToolResult) -> None:
        if self.memory_manager is None or self.session_id is None:
            return
        self._ensure_session()
        self.memory_manager.add_message(
            self.session_id,
            "system",
            json.dumps(result.to_dict(), sort_keys=True),
            timestamp=time(),
            metadata={"kind": "tool_result", "tool_name": result.tool_name, "session_id": self.session_id},
        )

    def _ensure_session(self) -> None:
        try:
            self.memory_manager.get_session(self.session_id)
        except KeyError:
            self.memory_manager.create_session(self.session_id)
