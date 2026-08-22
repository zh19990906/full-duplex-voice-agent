"""Asynchronous generation lifecycle manager."""

import inspect
from collections.abc import Awaitable, Callable, AsyncIterator
from time import time

from .session import GenerationSession, GenerationStatus


CancelHook = Callable[[GenerationSession], Awaitable[None] | None]


class GenerationManager:
    """Own generation session state without controlling model or TTS work."""

    def __init__(self) -> None:
        self._sessions: dict[str, GenerationSession] = {}
        self._active_session: GenerationSession | None = None
        self._cancel_hooks: list[CancelHook] = []

    @property
    def active_session(self) -> GenerationSession | None:
        """Return the currently active session, if one exists."""
        return self._active_session

    def add_cancel_hook(self, hook: CancelHook) -> None:
        """Register a hook notified after a session is cancelled."""
        self._cancel_hooks.append(hook)

    def remove_cancel_hook(self, hook: CancelHook) -> None:
        """Remove a previously registered cancellation hook."""
        try:
            self._cancel_hooks.remove(hook)
        except ValueError:
            return

    async def start_generation(self, generation_id: str | None = None) -> GenerationSession:
        """Create and activate a new running generation session."""
        if self._active_session is not None:
            await self.cancel_current()

        session = GenerationSession(id=generation_id) if generation_id else GenerationSession()
        session.status = GenerationStatus.RUNNING
        session.started_at = time()
        self._sessions[session.id] = session
        self._active_session = session
        return session

    async def cancel_current(self) -> GenerationSession | None:
        """Cancel only the active session and notify registered hooks."""
        session = self._active_session
        if session is None:
            return None

        session.status = GenerationStatus.CANCELLED
        session.cancelled_at = session.cancelled_at or time()
        self._active_session = None
        for hook in tuple(self._cancel_hooks):
            result = hook(session)
            if inspect.isawaitable(result):
                await result
        return session

    async def complete(self, session_id: str) -> GenerationSession:
        """Mark a known session complete and clear it if it is active."""
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown generation session: {session_id}") from exc

        if session.status is not GenerationStatus.CANCELLED:
            session.status = GenerationStatus.COMPLETED
            session.completed_at = session.completed_at or time()
            if self._active_session is session:
                self._active_session = None
        return session

    async def stream_tokens(self, session: GenerationSession) -> AsyncIterator[str]:
        """Reserve the future streaming boundary without generating tokens."""
        raise NotImplementedError("Token streaming is provided by a future LLM integration.")
