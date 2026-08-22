"""Tool registration and discovery."""

from __future__ import annotations

from collections.abc import Iterable

from .models import ToolDefinition


class ToolRegistry:
    """Own the set of tools available to one application composition."""

    def __init__(self, tools: Iterable[ToolDefinition] = ()) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools:
            self.register_tool(tool)

    def register_tool(
        self,
        tool: ToolDefinition | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, object] | None = None,
        handler=None,
    ) -> ToolDefinition:
        """Register a definition, supporting both object and keyword forms."""

        definition = tool or ToolDefinition(
            name=name or "",
            description=description or "",
            parameters=parameters or {},
            handler=handler,
        )
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = definition
        return definition

    def unregister_tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    register = register_tool
    unregister = unregister_tool

    def get_tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    lookup = get_tool

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    def discover_tools(self) -> tuple[dict[str, object], ...]:
        return tuple(tool.to_dict() for tool in self._tools.values())
