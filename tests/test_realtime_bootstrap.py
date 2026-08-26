import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.realtime.protocol import AudioFrameHeader, encode_audio_frame
from src.runtime_app.bootstrap import RealtimeServerSettings, build_realtime_app


class _FakeFastAPI:
    def __init__(self, **_kwargs):
        self.state = SimpleNamespace()
        self.http_endpoints = {}
        self.websocket_endpoints = {}

    def _route(self, method, path):
        def decorator(endpoint):
            if method == "websocket":
                self.websocket_endpoints[path] = endpoint
            else:
                self.http_endpoints[(method, path)] = endpoint
            return endpoint

        return decorator

    def get(self, path):
        return self._route("GET", path)

    def post(self, path):
        return self._route("POST", path)

    def delete(self, path):
        return self._route("DELETE", path)

    def websocket(self, path):
        return self._route("websocket", path)


class _Response:
    def __init__(self, body=None, status_code=200):
        self.body = body
        self.status_code = status_code


class _JSONResponse(_Response):
    pass


class _FileResponse(_Response):
    pass


class _FakeWebSocket:
    def __init__(self, path, *, bytes_messages=()):
        self.url = SimpleNamespace(path=path)
        self.query_params = {}
        self._messages = [{"bytes": value} for value in bytes_messages]
        self.accepted = False
        self.sent = []
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("disconnect")

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self, code=None, reason=None):
        self.closed = {"code": code, "reason": reason}


class RecordingRealtimeRuntime:
    def __init__(self):
        self.audio_sequences = []
        self.commands = []
        self.started = False
        self.closed = False
        self._events = []

    async def start(self):
        self.started = True

    async def accept_audio_frame(self, frame):
        self.audio_sequences.append(frame.sequence)
        self._events.append(
            {
                "event": "AUDIO_FRAME_ACCEPTED",
                "sequence": frame.sequence,
                "payload": {"sequence": frame.sequence},
            }
        )

    async def accept_command(self, command):
        self.commands.append(command)

    async def events(self):
        while self._events:
            yield self._events.pop(0)

    async def close(self):
        self.closed = True


def _fake_fastapi_modules():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeFastAPI
    fastapi.WebSocket = _FakeWebSocket
    fastapi.WebSocketDisconnect = RuntimeError

    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = _FileResponse
    responses.JSONResponse = _JSONResponse
    responses.Response = _Response
    return {"fastapi": fastapi, "fastapi.responses": responses}


class RealtimeBootstrapTests(unittest.TestCase):
    def test_settings_default_to_configurable_frontend_port_8001(self):
        settings = RealtimeServerSettings(static_dir=Path("frontend"))

        self.assertEqual(settings.port, 8001)

    def test_build_realtime_app_keeps_runtime_factory_on_app_state(self):
        settings = RealtimeServerSettings(static_dir=Path("frontend"))
        runtime = RecordingRealtimeRuntime()
        factory = lambda _session_id: runtime

        with patch.dict(sys.modules, _fake_fastapi_modules()):
            app = build_realtime_app(settings, runtime_factory=factory)

        self.assertIs(app.state.runtime_factory, factory)

    def test_session_delete_closes_runtime(self):
        runtime = RecordingRealtimeRuntime()
        settings = RealtimeServerSettings(static_dir=Path("frontend"))

        with patch.dict(sys.modules, _fake_fastapi_modules()):
            app = build_realtime_app(settings, runtime_factory=lambda _session_id: runtime)
            create_session = app.http_endpoints[("POST", "/sessions")]
            delete_session = app.http_endpoints[("DELETE", "/sessions/{session_id}")]
            created = asyncio.run(create_session())
            session_id = created.body["session_id"]
            response = asyncio.run(delete_session(session_id))

        self.assertEqual(response.status_code, 204)
        self.assertTrue(runtime.started)
        self.assertTrue(runtime.closed)

    def test_binary_audio_enters_runtime_and_event_returns(self):
        runtime = RecordingRealtimeRuntime()
        settings = RealtimeServerSettings(static_dir=Path("frontend"))
        frame = encode_audio_frame(
            AudioFrameHeader(sequence=0, capture_timestamp=0.0),
            b"\x00\x00" * 320,
        )

        with patch.dict(sys.modules, _fake_fastapi_modules()):
            app = build_realtime_app(settings, runtime_factory=lambda _session_id: runtime)
            create_session = app.http_endpoints[("POST", "/sessions")]
            websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
            session_id = asyncio.run(create_session()).body["session_id"]
            websocket = _FakeWebSocket(f"/ws/{session_id}", bytes_messages=[frame])
            asyncio.run(websocket_endpoint(websocket, session_id))

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent[0]["event"], "AUDIO_FRAME_ACCEPTED")
        self.assertEqual(runtime.audio_sequences, [0])


if __name__ == "__main__":
    unittest.main()
