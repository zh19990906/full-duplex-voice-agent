"""HTTP-facing application facade backed by the existing session layer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.session.manager import SessionManager

from .events import ApiEventSerializer
from .models import ApiResponse


class ApiService:
    """Route API requests without caching session or runtime state."""

    def __init__(self, session_manager: SessionManager, serializer: ApiEventSerializer | None = None) -> None:
        self.session_manager = session_manager
        self.serializer = serializer or ApiEventSerializer()

    async def handle(self, method: str, path: str, body: Any = None) -> ApiResponse:
        """Handle one decoded request and delegate state changes to SessionManager."""

        method = method.upper()
        parts = tuple(part for part in path.split("/") if part)
        document, error = self._decode_body(body)
        if error is not None:
            return error

        try:
            if method == "POST" and parts == ("sessions",):
                session = self.session_manager.create_session(
                    document.get("session_id"), document.get("metadata", {})
                )
                return ApiResponse(201, session.to_dict())

            if parts == ("sessions",) and method == "GET":
                return ApiResponse(200, {"sessions": [session.to_dict() for session in self.session_manager.list_sessions()]})

            if len(parts) == 2 and parts[0] == "sessions":
                session_id = parts[1]
                session = self.session_manager.get_session(session_id)
                if method == "GET":
                    return ApiResponse(200, session.to_dict())
                if method == "DELETE":
                    self.session_manager.close_session(session_id)
                    self.session_manager.delete_session(session_id)
                    return ApiResponse(204)

            if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "message" and method == "POST":
                session = self.session_manager.get_session(parts[1])
                message = document.get("message")
                if not isinstance(message, str) or not message.strip():
                    return ApiResponse(400, {"error": "message must be a non-empty string"})
                response = await session.agent.run(message)
                return ApiResponse(200, {"session_id": parts[1], "response": response})
        except KeyError as exc:
            return ApiResponse(404, {"error": str(exc)})
        except ValueError as exc:
            return ApiResponse(400, {"error": str(exc)})
        except RuntimeError as exc:
            return ApiResponse(409, {"error": str(exc)})

        return ApiResponse(404, {"error": "route not found"})

    @staticmethod
    def _decode_body(body: Any) -> tuple[dict[str, Any], ApiResponse | None]:
        if body is None:
            return {}, None
        if isinstance(body, Mapping):
            return dict(body), None
        if isinstance(body, (bytes, str)):
            try:
                decoded = json.loads(body)
            except (TypeError, ValueError) as exc:
                return {}, ApiResponse(400, {"error": f"invalid JSON body: {exc}"})
            if isinstance(decoded, Mapping):
                return dict(decoded), None
        return {}, ApiResponse(400, {"error": "request body must be a JSON object"})
