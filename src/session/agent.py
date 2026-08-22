"""Session-bound application agent execution."""

from __future__ import annotations

import asyncio
from typing import Any

from src.agent.loop import AgentLoop
from src.agent.models import AgentState
from src.application.voice_agent import VoiceAgent
from src.memory.manager import MemoryManager

from .models import Session, SessionStatus


class SessionAgent:
    """Bind one VoiceAgent and AgentLoop to one isolated session."""

    def __init__(
        self,
        session: Session,
        memory_manager: MemoryManager,
        *,
        voice_agent: VoiceAgent | None = None,
        agent_loop: AgentLoop | None = None,
    ) -> None:
        if session.memory_manager is not None and session.memory_manager is not memory_manager:
            raise ValueError("session memory does not match SessionAgent memory")
        self.session = session
        self.memory_manager = memory_manager
        self.voice_agent = voice_agent
        self.agent_loop = agent_loop
        self._lock = asyncio.Lock()

    @property
    def state(self) -> AgentState:
        if self.agent_loop is None:
            return AgentState(session_id=self.session.session_id)
        return self.agent_loop.state

    async def run(self, user_input: str) -> str:
        """Run only this session's agent, serializing same-session requests."""

        async with self._lock:
            if self.session.status is SessionStatus.CREATED:
                self.session.activate()
            if self.session.status is SessionStatus.IDLE:
                self.session.activate()
            if self.session.status is not SessionStatus.ACTIVE:
                raise RuntimeError(f"session is not executable: {self.session.status.value}")
            try:
                if self.voice_agent is not None:
                    return await self.voice_agent.run_agent(user_input)
                if self.agent_loop is not None:
                    return await self.agent_loop.run(user_input)
                raise RuntimeError("session agent has no VoiceAgent or AgentLoop")
            finally:
                if self.session.status is SessionStatus.ACTIVE:
                    self.session.mark_idle()
