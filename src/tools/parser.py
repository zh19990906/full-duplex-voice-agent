"""Provider-neutral parsing of structured native LLM tool calls."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from src.agent.models import ToolCallRequest
from .registry import ToolRegistry


class ToolCallFormat(ABC):
    """Extract a generic tool-call mapping from one provider-neutral shape."""

    @abstractmethod
    def extract(self, value: Mapping[str, Any]) -> Mapping[str, Any] | None:
        raise NotImplementedError


class GenericToolCallFormat(ToolCallFormat):
    """Support direct and ``tool_call``-wrapped JSON objects."""

    def extract(self, value: Mapping[str, Any]) -> Mapping[str, Any] | None:
        nested = value.get("tool_call")
        if isinstance(nested, Mapping):
            function = nested.get("function")
            if isinstance(function, Mapping):
                return function
            return nested
        if ("tool" in value or "tool_name" in value or "name" in value) and "arguments" in value:
            return value
        return None


class ToolCallParser:
    """Convert generic JSON or mappings to existing ``ToolCallRequest`` values."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        formats: tuple[ToolCallFormat, ...] | None = None,
    ) -> None:
        self.registry = registry
        self.formats = formats or (GenericToolCallFormat(),)

    def parse(self, output: Any) -> ToolCallRequest | None:
        """Return a tool request, ``None`` for final text, or raise malformed-input errors."""

        if isinstance(output, ToolCallRequest):
            self._validate_tool(output.tool_name)
            return output

        value = output
        if isinstance(output, str):
            stripped = output.strip()
            if not stripped or not stripped.startswith(("{", "[")):
                return None
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON tool call: {exc.msg}") from exc

        if not isinstance(value, Mapping):
            return None
        for tool_format in self.formats:
            candidate = tool_format.extract(value)
            if candidate is not None:
                request = self._request_from_mapping(candidate)
                self._validate_tool(request.tool_name)
                return request
        return None

    @staticmethod
    def _request_from_mapping(candidate: Mapping[str, Any]) -> ToolCallRequest:
        normalized = dict(candidate)
        arguments = normalized.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON tool arguments: {exc.msg}") from exc
        if not isinstance(arguments, Mapping):
            raise ValueError("tool call arguments must be an object")
        normalized["arguments"] = dict(arguments)
        return ToolCallRequest.from_mapping(normalized)

    def _validate_tool(self, name: str) -> None:
        if self.registry is None:
            return
        try:
            self.registry.get_tool(name)
        except KeyError as exc:
            raise ValueError(f"unknown tool: {name}") from exc
