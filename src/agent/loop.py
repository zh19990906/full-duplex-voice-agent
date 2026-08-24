"""Bounded ReAct-style application orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from time import time

from src.memory.context import ContextBuilder
from src.memory.manager import MemoryManager
from src.memory.models import ConversationMessage
from src.tools.models import ToolResult
from src.tools.parser import ToolCallParser
from src.tools.router import ToolRouter

from .models import AgentState, AgentStatus, ToolCallRequest


AgentDecision = str | ToolCallRequest | Mapping[str, Any]
DecisionProvider = Callable[[tuple[ConversationMessage, ...]], Awaitable[AgentDecision] | AgentDecision]


class AgentLoopError(RuntimeError):
    """Raised when an agent run cannot produce a final response."""


class AgentLoop:
    """Coordinate context, model decisions, tools, observations, and memory."""

    def __init__(
        self,
        decision_provider: DecisionProvider,
        *,
        memory_manager: MemoryManager,
        tool_router: ToolRouter,
        session_id: str,
        context_builder: ContextBuilder | None = None,
        max_iterations: int = 5,
        tool_timeout: float | None = None,
        tool_parser: ToolCallParser | None = None,
    ) -> None:
        if not callable(decision_provider):
            raise TypeError("decision_provider must be callable")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        self.decision_provider = decision_provider
        self.memory_manager = memory_manager
        self.tool_router = tool_router
        self.session_id = session_id
        self.context_builder = context_builder or ContextBuilder(memory_manager)
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.tool_parser = tool_parser or ToolCallParser()
        self.state = AgentState(session_id=session_id)
        self.state_history: list[AgentStatus] = [AgentStatus.IDLE]

    async def run(self, user_input: str) -> str:
        """Run a bounded thought/tool/observation loop and return final text."""

        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must not be empty")
        self._ensure_session()
        self.state = AgentState(session_id=self.session_id)
        self.state_history = [AgentStatus.IDLE]
        self._remember("user", user_input, {"kind": "user_request"})

        try:
            for iteration in range(1, self.max_iterations + 1):
                self.state.iteration = iteration
                self._set_status(AgentStatus.THINKING)
                context = self.context_builder.build(self.session_id)
                decision = await self._request_decision(context)
                tool_call = self._normalize_tool_call(decision)
                if tool_call is None:
                    response = self._normalize_final_response(decision)
                    self._remember("assistant", response, {"kind": "final_response"})
                    self._set_status(AgentStatus.COMPLETED)
                    return response

                self._set_status(AgentStatus.EXECUTING_TOOL)
                result = await self._execute_tool(tool_call)
                self._set_status(AgentStatus.OBSERVING)
                self._remember_observation(tool_call, result)

            raise AgentLoopError(f"agent exceeded maximum iterations: {self.max_iterations}")
        except AgentLoopError:
            self._set_status(AgentStatus.FAILED)
            raise
        except Exception as exc:
            self._set_status(AgentStatus.FAILED)
            raise AgentLoopError(f"agent loop failed: {exc}") from exc

    def _set_status(self, status: AgentStatus) -> None:
        self.state.status = status
        self.state_history.append(status)

    async def _request_decision(self, context: tuple[ConversationMessage, ...]) -> AgentDecision:
        decision = self.decision_provider(context)
        if hasattr(decision, "__await__"):
            decision = await decision
        return decision

    async def _execute_tool(self, request: ToolCallRequest) -> ToolResult:
        route = self.tool_router.route(
            {"name": request.name, "arguments": dict(request.arguments)}
        )
        if self.tool_timeout is None:
            return await route
        try:
            return await asyncio.wait_for(route, timeout=self.tool_timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"tool execution timed out after {self.tool_timeout:g}s",
                tool_name=request.name,
            )

    def _normalize_tool_call(self, decision: AgentDecision) -> ToolCallRequest | None:
        if isinstance(decision, ToolCallRequest):
            return decision
        return self.tool_parser.parse(decision)

    @staticmethod
    def _normalize_final_response(decision: AgentDecision) -> str:
        if isinstance(decision, str):
            return decision
        if isinstance(decision, Mapping):
            text = decision.get("text")
            if isinstance(text, str):
                return text
        raise AgentLoopError("LLM decision must be final text or a tool call")

    def _remember(self, role: str, content: str, metadata: Mapping[str, Any]) -> None:
        message = self.memory_manager.add_message(
            self.session_id,
            role,
            content,
            timestamp=time(),
            metadata={"session_id": self.session_id, **dict(metadata)},
        )
        self.state.messages.append(message.to_dict())

    def _remember_observation(self, request: ToolCallRequest, result: ToolResult) -> None:
        self._remember(
            "system",
            json.dumps(
                {"call_id": request.call_id, "tool": request.name, "result": result.to_dict()},
                sort_keys=True,
            ),
            {"kind": "observation", "tool_name": request.name, "call_id": request.call_id},
        )

    def _ensure_session(self) -> None:
        try:
            self.memory_manager.get_session(self.session_id)
        except KeyError:
            self.memory_manager.create_session(self.session_id)
