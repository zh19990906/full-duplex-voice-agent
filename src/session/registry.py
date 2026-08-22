"""Thread-safe registry for active application sessions."""

from __future__ import annotations

from threading import RLock

from .models import Session


class SessionRegistry:
    """Own session lookup without owning session internals."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def register(self, session: Session) -> None:
        with self._lock:
            if session.session_id in self._sessions:
                raise ValueError(f"session already exists: {session.session_id}")
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def remove(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions.pop(session_id)
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def list(self) -> tuple[Session, ...]:
        with self._lock:
            return tuple(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
