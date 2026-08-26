"""Application startup entry points for legacy app bootstrap and realtime server."""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler
from src.realtime.audio_ingress import RealtimeAudioFrame
from src.realtime.protocol import decode_audio_frame
from src.api.events import ApiEventSerializer

from .container import ApplicationContainer, SlowConsumerError


def create_application(
    *,
    event_bus: EventBus | None = None,
    controller: ConversationController | None = None,
    generation_manager: GenerationManager | None = None,
    scheduler: RealtimeScheduler | None = None,
) -> ApplicationContainer:
    """Create an uninitialized application container."""
    return ApplicationContainer(
        event_bus=event_bus,
        controller=controller,
        generation_manager=generation_manager,
        scheduler=scheduler,
    )


async def initialize_application(
    *,
    event_bus: EventBus | None = None,
    controller: ConversationController | None = None,
    generation_manager: GenerationManager | None = None,
    scheduler: RealtimeScheduler | None = None,
) -> ApplicationContainer:
    """Create and initialize an application container."""
    application = create_application(
        event_bus=event_bus,
        controller=controller,
        generation_manager=generation_manager,
        scheduler=scheduler,
    )
    await application.initialize()
    return application


@dataclass(frozen=True)
class RealtimeServerSettings:
    """Stable server settings for the realtime app composition root."""

    static_dir: Path
    host: str = "0.0.0.0"
    port: int = 8001


def build_realtime_app(
    settings: RealtimeServerSettings,
    *,
    runtime_factory: Any,
):
    """Build the realtime websocket app around a per-session runtime factory."""
    if not isinstance(settings, RealtimeServerSettings):
        raise TypeError("settings must be RealtimeServerSettings")
    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable")

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, Response
    globals()["WebSocket"] = WebSocket
    globals()["WebSocketDisconnect"] = WebSocketDisconnect

    sessions: dict[str, Any] = {}
    static_dir = settings.static_dir.resolve()

    @asynccontextmanager
    async def lifespan(app: Any):
        try:
            yield
        finally:
            for runtime in tuple(sessions.values()):
                await _safe_close_runtime(runtime)
            sessions.clear()
            factory_close = getattr(runtime_factory, "close", None)
            if callable(factory_close):
                result = factory_close()
                if inspect.isawaitable(result):
                    await result

    app = FastAPI(title="Full Duplex Voice Agent", lifespan=lifespan)
    if not hasattr(app, "state"):
        app.state = SimpleNamespace()
    app.state.runtime_factory = runtime_factory
    app.state.sessions = sessions

    @app.get("/health")
    async def health():
        return {"status": "ok", "port": settings.port, "sessions": len(sessions)}

    @app.post("/sessions")
    async def create_session():
        session_id = uuid.uuid4().hex
        runtime = runtime_factory(session_id)
        await _maybe_call(runtime, "start")
        sessions[session_id] = runtime
        return JSONResponse({"session_id": session_id, "status": "CREATED"}, status_code=201)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        runtime = sessions.get(session_id)
        if runtime is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"session_id": session_id, "status": "ACTIVE"})

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        runtime = sessions.pop(session_id, None)
        if runtime is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        await _safe_close_runtime(runtime)
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/message")
    async def send_message(session_id: str, body: dict[str, Any]):
        runtime = sessions.get(session_id)
        message = body.get("message")
        if runtime is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        if not isinstance(message, str) or not message.strip():
            return JSONResponse({"error": "message must be a non-empty string"}, status_code=400)
        await runtime.accept_command({"type": "text", "text": message})
        return JSONResponse({"session_id": session_id, "accepted": True}, status_code=202)

    async def handle_websocket(websocket: WebSocket, session_id: str | None = None):
        await websocket.accept()
        resolved_session_id = session_id or websocket.query_params.get("session_id")
        runtime = sessions.get(resolved_session_id or "")
        if runtime is None:
            await websocket.send_json(
                {"event": "error", "payload": {"message": "unknown session"}}
            )
            await websocket.close(code=1008, reason="unknown session")
            return

        await _maybe_call(runtime, "connect")
        sender = asyncio.create_task(_forward_runtime_events(runtime, websocket, resolved_session_id))
        sender_failure: BaseException | None = None
        try:
            while True:
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    break
                try:
                    await _handle_socket_message(runtime, message)
                    if sender.done():
                        try:
                            await sender
                        except SlowConsumerError as exc:
                            sender_failure = exc
                            break
                        sender = asyncio.create_task(
                            _forward_runtime_events(runtime, websocket, resolved_session_id)
                        )
                    await asyncio.sleep(0)
                except Exception as exc:
                    await websocket.send_json(
                        {"event": "error", "payload": {"message": str(exc)}}
                    )
        finally:
            if sender.done():
                try:
                    await sender
                except SlowConsumerError as exc:
                    sender_failure = exc
                except asyncio.CancelledError:
                    pass
            else:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass
            if isinstance(sender_failure, SlowConsumerError):
                sessions.pop(resolved_session_id or "", None)
                await _safe_close_runtime(runtime)
                await websocket.close(code=1011, reason="slow consumer")
            try:
                await _maybe_call(runtime, "disconnect")
            finally:
                if isinstance(sender_failure, SlowConsumerError):
                    return

    @app.websocket("/ws/{session_id}")
    async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
        await handle_websocket(websocket, session_id)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await handle_websocket(websocket)

    @app.get("/{asset_path:path}")
    async def static_files(asset_path: str):
        relative = asset_path or "index.html"
        path = (static_dir / relative).resolve()
        if static_dir not in path.parents and path != static_dir:
            return JSONResponse({"error": "invalid path"}, status_code=400)
        if not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    return app


async def _forward_runtime_events(runtime: Any, websocket: Any, session_id: str | None) -> None:
    async for event in runtime.events():
        await websocket.send_json(ApiEventSerializer.serialize(event, session_id))


async def _handle_socket_message(runtime: Any, message: dict[str, Any]) -> None:
    text = message.get("text")
    if text:
        document = json.loads(text)
        if not isinstance(document, dict):
            raise ValueError("JSON websocket messages must be objects")
        await runtime.accept_command(document)
        return
    frame = message.get("bytes")
    if frame is not None:
        header, pcm = decode_audio_frame(frame)
        await runtime.accept_audio_frame(RealtimeAudioFrame(header=header, pcm=pcm))


async def _maybe_call(value: Any, name: str) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return None
    result = method()
    if inspect.isawaitable(result):
        return await result
    return result


async def _safe_close_runtime(runtime: Any) -> None:
    close = getattr(runtime, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result
