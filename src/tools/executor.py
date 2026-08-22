"""Isolated, timeout-bounded tool execution."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Mapping

from .models import ToolDefinition, ToolResult


class ToolExecutor:
    """Execute sync or async handlers without leaking tool exceptions."""

    def __init__(self, default_timeout: float = 5.0) -> None:
        if default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        self.default_timeout = default_timeout

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, Any],
        timeout: float | None = None,
    ) -> ToolResult:
        limit = self.default_timeout if timeout is None else timeout
        if limit <= 0:
            raise ValueError("timeout must be positive")
        try:
            result = await asyncio.wait_for(self._invoke(tool, arguments), timeout=limit)
            return ToolResult(success=True, output=str(result), tool_name=tool.name)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"tool execution timed out after {limit:g}s",
                tool_name=tool.name,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool_name=tool.name)

    async def _invoke(self, tool: ToolDefinition, arguments: Mapping[str, Any]) -> Any:
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**dict(arguments))
        return await asyncio.to_thread(tool.handler, **dict(arguments))
