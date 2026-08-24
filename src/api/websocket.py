"""Framework-neutral asynchronous event and audio streaming boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.session.manager import SessionManager

from .events import ApiEventSerializer


AudioHandler = Callable[[Any, bytes], Awaitable[None] | None]


class WebSocketEventConnection:
    """One transport connection bound to an existing SessionAgent."""

    def __init__(self, session: Any, serializer: ApiEventSerializer, audio_handler: AudioHandler | None = None) -> None:
        self.session = session
        self.serializer = serializer
        self.audio_handler = audio_handler
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False

    async def publish(self, value: Any) -> None:
        if self._closed:
            raise RuntimeError("WebSocket connection is closed")
        await self._events.put(self.serializer.serialize(value, self.session.session_id))

    async def receive_event(self) -> dict[str, Any]:
        if self._closed and self._events.empty():
            raise RuntimeError("WebSocket connection is closed")
        return await self._events.get()

    async def receive(self, message: Mapping[str, Any]) -> None:
        """Accept text or raw audio input without decoding audio payloads."""

        if self._closed:
            raise RuntimeError("WebSocket connection is closed")
        message_type = message.get("type")
        if message_type == "audio":
            payload = message.get("data")
            if not isinstance(payload, bytes):
                raise ValueError("audio data must be bytes at the existing audio boundary")
            if self.audio_handler is None:
                raise RuntimeError("audio handler is not configured")
            result = self.audio_handler(self.session, payload)
            if hasattr(result, "__await__"):
                await result
            return
        if message_type in {"text", "message"}:
            content = message.get("text", message.get("message"))
            if not isinstance(content, str) or not content.strip():
                raise ValueError("text message must be a non-empty string")
            response = await self.session.agent.run(content)
            await self.publish({"event": "text_response", "payload": {"text": response}})
            return
        raise ValueError(f"unsupported WebSocket message type: {message_type!r}")

    async def close(self) -> None:
        self._closed = True


class WebSocketEndpoint:
    """Create transport connections while delegating session lookup."""

    def __init__(self, session_manager: SessionManager, serializer: ApiEventSerializer | None = None, audio_handler: AudioHandler | None = None) -> None:
        self.session_manager = session_manager
        self.serializer = serializer or ApiEventSerializer()
        self.audio_handler = audio_handler

    async def connect(self, session_id: str) -> WebSocketEventConnection:
        session = self.session_manager.get_session(session_id)
        return WebSocketEventConnection(session, self.serializer, self.audio_handler)
