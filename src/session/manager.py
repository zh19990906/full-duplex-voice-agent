"""Session lifecycle coordinator and per-session dependency composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4

from src.memory.manager import MemoryManager

from .agent import SessionAgent
from .models import Session, SessionStatus
from .registry import SessionRegistry


SessionAgentFactory = Callable[[Session, MemoryManager], SessionAgent]


class SessionManager:
    """Create isolated sessions and manage their lifecycle."""

    def __init__(
        self,
        *,
        max_sessions: int = 100,
        registry: SessionRegistry | None = None,
        agent_factory: SessionAgentFactory | None = None,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self.registry = registry or SessionRegistry()
        self.agent_factory = agent_factory
        self._lock = RLock()

    def create_session(
        self,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        with self._lock:
            if len(self.registry) >= self.max_sessions:
                raise RuntimeError("maximum session limit reached")
            session_id = session_id or uuid4().hex
            memory = MemoryManager()
            memory.create_session(session_id)
            session = Session(session_id=session_id, created_at=time(), metadata=dict(metadata or {}))
            session.memory_manager = memory
            session.add_resource("memory")
            if self.agent_factory is None:
                session.agent = SessionAgent(session, memory)
            else:
                session.agent = self.agent_factory(session, memory)
            session.add_resource("agent")
            self.registry.register(session)
            return session

    def get_session(self, session_id: str) -> Session:
        return self.registry.get(session_id)

    def activate_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        session.activate()
        return session

    def mark_idle(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        session.mark_idle()
        return session

    def close_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        session.begin_closing()
        session.close()
        return session

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session.status not in {SessionStatus.CLOSED, SessionStatus.FAILED}:
            raise ValueError(f"session must be closed before deletion: {session_id}")
        if session.memory_manager is not None:
            session.memory_manager.delete_session(session_id)
        self.registry.remove(session_id)

    def list_sessions(self) -> tuple[Session, ...]:
        return self.registry.list()
