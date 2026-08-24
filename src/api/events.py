"""Convert runtime/application values into client event envelopes."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from src.agent.models import AgentState, ToolCallRequest
from src.core.events.events import BaseEvent
from src.llm_runtime.stream import TokenChunk
from src.tools.models import ToolResult
from src.tts_runtime.stream import AudioChunk


class ApiEventSerializer:
    """Stateless serializer that leaves source contracts unchanged."""

    @staticmethod
    def serialize(value: Any, session_id: str | None = None) -> dict[str, Any]:
        if isinstance(value, BaseEvent):
            event_document = value.to_dict()
            event_name = event_document["event"]
            payload = event_document["payload"]
        elif isinstance(value, TokenChunk):
            event_name = "token"
            payload = value.to_dict()
        elif isinstance(value, AudioChunk):
            event_name = "audio_chunk"
            payload = value.to_dict()
        elif isinstance(value, AgentState):
            event_name = "agent_state"
            payload = value.to_dict()
        elif isinstance(value, ToolCallRequest):
            event_name = "tool_call"
            payload = value.to_dict()
        elif isinstance(value, ToolResult):
            event_name = "tool_result"
            payload = value.to_dict()
        elif isinstance(value, Mapping):
            event_name = str(value.get("event", "message"))
            payload = dict(value.get("payload", value))
        else:
            raise TypeError(f"unsupported API event value: {type(value).__name__}")

        envelope: dict[str, Any] = {
            "event": event_name,
            "payload": ApiEventSerializer._json_safe(payload),
        }
        if isinstance(value, BaseEvent):
            envelope.update(
                {
                    "event_id": event_document["event_id"],
                    "timestamp": event_document["timestamp"],
                    "source": event_document["source"],
                }
            )
        if session_id is not None:
            envelope["session_id"] = session_id
        return envelope

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, Mapping):
            return {str(key): ApiEventSerializer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ApiEventSerializer._json_safe(item) for item in value]
        return value
