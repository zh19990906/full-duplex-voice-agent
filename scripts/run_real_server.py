"""Serve the real text-to-speech path and bundled frontend on port 8001."""

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_app(args: argparse.Namespace):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, Response
    from src.adapters.llm.providers.qwen_transformers import TransformersQwenProvider
    from src.adapters.tts.providers.cosyvoice_worker import CosyVoiceWorkerClient
    from src.api.events import ApiEventSerializer
    from src.runtime_app.real_session import RealModelSession

    llm = TransformersQwenProvider(args.llm_model, device="cuda")
    tts = CosyVoiceWorkerClient(
        args.tts_model,
        worker_python=args.worker_python,
        worker_script=args.worker_script,
        cosyvoice_root=args.cosy_root,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        startup_timeout=args.worker_startup_timeout,
    )
    sessions: dict[str, SimpleNamespace] = {}
    model_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await tts.close()

    app = FastAPI(title="Full Duplex Voice Agent", lifespan=lifespan)

    def prompt_builder(text: str) -> str:
        return (
            "请回答用户的问题，回答要自然、简洁，适合直接转换成语音。\n"
            f"用户：{text}\n助手："
        )

    async def publish(session_id: str, sockets: set[WebSocket], value) -> None:
        envelope = ApiEventSerializer.serialize(value, session_id)
        stale = []
        for socket in tuple(sockets):
            try:
                await socket.send_json(envelope)
            except Exception:
                stale.append(socket)
        sockets.difference_update(stale)

    def create_state(session_id: str) -> SimpleNamespace:
        sockets: set[WebSocket] = set()
        session = SimpleNamespace(session_id=session_id, sockets=sockets)
        session.model = RealModelSession(
            llm,
            tts,
            lambda value: publish(session_id, sockets, value),
            prompt_builder=prompt_builder,
        )
        return session

    @app.get("/health")
    async def health():
        return {"status": "ok", "port": args.port, "sessions": len(sessions)}

    @app.post("/sessions")
    async def create_session():
        session_id = uuid.uuid4().hex
        sessions[session_id] = create_state(session_id)
        return JSONResponse({"session_id": session_id, "status": "CREATED"}, status_code=201)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = sessions.pop(session_id, None)
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        await session.model.interrupt()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/message")
    async def send_message(session_id: str, body: dict):
        session = sessions.get(session_id)
        message = body.get("message")
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        if not isinstance(message, str) or not message.strip():
            return JSONResponse({"error": "message must be a non-empty string"}, status_code=400)
        try:
            async with model_lock:
                response = await session.model.run(message)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        await publish(session_id, session.sockets, {"event": "text_response", "payload": {"text": response}})
        return {"session_id": session_id, "response": response}

    async def handle_websocket(websocket: WebSocket, session_id: str | None = None):
        print(
            f"REAL_WS_HANDLER_ENTERED path={websocket.url.path} session={session_id!r}",
            flush=True,
        )
        await websocket.accept()
        print("REAL_WS_ACCEPTED", flush=True)
        session_id = session_id or websocket.query_params.get("session_id")
        session = sessions.get(session_id or "")
        if session is None:
            await websocket.send_json({
                "event": "error",
                "payload": {"message": "unknown session"},
            })
            await websocket.close(code=1008, reason="unknown session")
            return
        session.sockets.add(websocket)
        try:
            while True:
                message = await websocket.receive()
                if message.get("text"):
                    document = json.loads(message["text"])
                    if document.get("type") in {"text", "message"}:
                        text = document.get("text", document.get("message"))
                        if isinstance(text, str) and text.strip():
                            async with model_lock:
                                response = await session.model.run(text)
                            await publish(session_id, session.sockets, {"event": "text_response", "payload": {"text": response}})
                    elif document.get("type") == "interrupt":
                        await session.model.interrupt()
                elif message.get("bytes") is not None:
                    await websocket.send_json({
                        "event": "error",
                        "payload": {
                            "message": "实时浏览器音频尚未接入 X2-Turn 离线推理；请先使用文字输入。",
                        },
                    })
        except WebSocketDisconnect:
            pass
        finally:
            session.sockets.discard(websocket)

    @app.websocket("/ws/{session_id}")
    async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
        await handle_websocket(websocket, session_id)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # Keep the query-string form for older clients; the frontend uses the
        # explicit path form because some deployments do not route this form.
        await handle_websocket(websocket)

    frontend = ROOT / "frontend"

    @app.get("/{asset_path:path}")
    async def static_files(asset_path: str):
        path = (frontend / (asset_path or "index.html")).resolve()
        if frontend not in path.parents and path != frontend:
            return JSONResponse({"error": "invalid path"}, status_code=400)
        if not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real frontend backend on port 8001")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--worker-startup-timeout", type=float, default=180.0)
    parser.add_argument("--worker-python", default="/home/CosyVoice/.venv/bin/python")
    parser.add_argument("--cosy-root", default="/home/CosyVoice")
    parser.add_argument(
        "--worker-script",
        default=str(Path(__file__).with_name("cosyvoice_worker.py")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
