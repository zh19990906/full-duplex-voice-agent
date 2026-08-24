"""Session and conversation persistence abstractions."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from src.memory.models import ConversationMessage, ConversationState, message_from_mapping
from src.memory.store import MemoryStore

from .models import Session, SessionStatus


class SessionStore(ABC):
    """Persistence contract owned by the application/session boundary."""

    @abstractmethod
    def save_session(self, session: Session) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_session(self, session_id: str) -> Session:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> tuple[Session, ...]:
        raise NotImplementedError


class SQLiteSessionStore(SessionStore, MemoryStore):
    """SQLite-backed session store that reuses the existing MemoryStore contract."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    agent_state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_states (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    user_preferences_json TEXT NOT NULL,
                    current_state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )

    def save_session(self, session: Session) -> None:
        agent_state = dict(session.agent_state)
        if session.agent is not None:
            state = getattr(session.agent, "state", None)
            if state is not None and hasattr(state, "to_dict"):
                agent_state = state.to_dict()
                session.agent_state = dict(agent_state)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions(session_id, created_at, status, metadata_json, agent_state_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    agent_state_json=excluded.agent_state_json
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.status.value,
                    json.dumps(session.metadata, sort_keys=True),
                    json.dumps(agent_state, sort_keys=True),
                ),
            )

    def load_session(self, session_id: str) -> Session:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        return Session(
            session_id=row["session_id"],
            created_at=row["created_at"],
            status=SessionStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
            agent_state=json.loads(row["agent_state_json"]),
        )

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM conversation_states WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def list_sessions(self) -> tuple[Session, ...]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM sessions ORDER BY created_at, session_id").fetchall()
        return tuple(self._row_to_session(row) for row in rows)

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            session_id=row["session_id"],
            created_at=row["created_at"],
            status=SessionStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
            agent_state=json.loads(row["agent_state_json"]),
        )

    def append_message(self, session_id: str, message: ConversationMessage) -> None:
        self.get_state(session_id)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversation_messages(session_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, message.role, message.content, message.timestamp, json.dumps(message.metadata, sort_keys=True)),
            )

    def get_messages(self, session_id: str) -> tuple[ConversationMessage, ...]:
        self.get_state(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, content, timestamp, metadata_json
                FROM conversation_messages
                WHERE session_id = ? ORDER BY message_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(
            message_from_mapping(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            )
            for row in rows
        )

    def save_state(self, state: ConversationState) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversation_states(session_id, summary, user_preferences_json, current_state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary=excluded.summary,
                    user_preferences_json=excluded.user_preferences_json,
                    current_state_json=excluded.current_state_json
                """,
                (
                    state.session_id,
                    state.summary,
                    json.dumps(state.user_preferences, sort_keys=True),
                    json.dumps(state.current_state, sort_keys=True),
                ),
            )

    def get_state(self, session_id: str) -> ConversationState:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM conversation_states WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown memory session: {session_id}")
        return ConversationState(
            session_id=row["session_id"],
            summary=row["summary"],
            user_preferences=json.loads(row["user_preferences_json"]),
            current_state=json.loads(row["current_state_json"]),
        )

    def clear_messages(self, session_id: str) -> None:
        self.get_state(session_id)
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))

    def close(self) -> None:
        with self._lock:
            self._connection.close()
