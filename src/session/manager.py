"""Session lifecycle coordinator and per-session dependency composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4

from src.memory.manager import MemoryManager
from src.memory.store import InMemoryStore, MemoryStore

from .agent import SessionAgent
from .models import Session, SessionStatus
from .registry import SessionRegistry
from .storage import SessionStore


SessionAgentFactory = Callable[[Session, MemoryManager], SessionAgent]


class SessionManager:
    """Create isolated sessions and manage their lifecycle."""

    def __init__(
        self,
        *,
        max_sessions: int = 100,
        registry: SessionRegistry | None = None,
        agent_factory: SessionAgentFactory | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self.registry = registry or SessionRegistry()
        self.agent_factory = agent_factory
        self.session_store = session_store
        self._shared_memory_store = session_store if isinstance(session_store, MemoryStore) else None
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
            if self.session_store is not None:
                try:
                    self.session_store.load_session(session_id)
                except KeyError:
                    pass
                else:
                    raise ValueError(f"session already exists: {session_id}")
            memory = self._new_memory()
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
            self._save_session(session)
            return session

    def get_session(self, session_id: str) -> Session:
        try:
            return self.registry.get(session_id)
        except KeyError:
            if self.session_store is None:
                raise
            stored = self.session_store.load_session(session_id)
            session = self._attach_runtime(stored)
            self.registry.register(session)
            return session

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
        self._save_session(session)
        return session

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session.status not in {SessionStatus.CLOSED, SessionStatus.FAILED}:
            raise ValueError(f"session must be closed before deletion: {session_id}")
        if session.memory_manager is not None:
            session.memory_manager.delete_session(session_id)
        if self.session_store is not None:
            self.session_store.delete_session(session_id)
        self.registry.remove(session_id)

    def list_sessions(self) -> tuple[Session, ...]:
        if self.session_store is not None:
            for stored in self.session_store.list_sessions():
                try:
                    self.registry.get(stored.session_id)
                except KeyError:
                    self.registry.register(self._attach_runtime(stored))
        return self.registry.list()

    def _new_memory(self) -> MemoryManager:
        if self._shared_memory_store is not None:
            return MemoryManager(store=self._shared_memory_store)
        return MemoryManager(store=InMemoryStore())

    def _attach_runtime(self, session: Session) -> Session:
        memory = self._new_memory()
        memory.get_session(session.session_id)
        session.memory_manager = memory
        if self.agent_factory is None:
            session.agent = SessionAgent(session, memory)
        else:
            session.agent = self.agent_factory(session, memory)
        session.add_resource("memory")
        session.add_resource("agent")
        self._restore_agent_state(session)
        return session

    def _save_session(self, session: Session) -> None:
        if self.session_store is None:
            return
        state = getattr(session.agent, "state", None)
        if state is not None and hasattr(state, "to_dict"):
            session.agent_state = state.to_dict()
        self.session_store.save_session(session)

    @staticmethod
    def _restore_agent_state(session: Session) -> None:
        if not session.agent_state or session.agent is None:
            return
        loop = getattr(session.agent, "agent_loop", None)
        state = getattr(loop, "state", None)
        if state is None:
            return
        from src.agent.models import AgentStatus

        state.iteration = int(session.agent_state.get("iteration", 0))
        state.status = AgentStatus(session.agent_state.get("status", AgentStatus.IDLE.value))
        state.messages = list(session.agent_state.get("messages", []))
