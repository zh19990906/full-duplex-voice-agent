import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.realtime.protocol import AudioFrameHeader, encode_audio_frame
from src.runtime_app.bootstrap import RealtimeServerSettings, build_realtime_app
from src.runtime_app.container import SlowConsumerError


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


class _FakeWebSocketDisconnect(Exception):
    pass


class _FakeWebSocket:
    def __init__(self, path, *, query_params=None, messages=()):
        self.url = SimpleNamespace(path=path)
        self.query_params = dict(query_params or {})
        self._messages = list(messages)
        self.accepted = False
        self.sent = []
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        raise _FakeWebSocketDisconnect()

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
        self.connected = 0
        self.disconnected = 0
        self._events = []

    async def start(self):
        self.started = True

    async def connect(self):
        self.connected += 1

    async def disconnect(self):
        self.disconnected += 1

    async def accept_audio_frame(self, frame):
        self.audio_sequences.append(frame.sequence)
        self._events.append({"event": "AUDIO_FRAME_ACCEPTED", "payload": {"sequence": frame.sequence}})

    async def accept_command(self, command):
        self.commands.append(command)
        if command.get("type") == "text":
            self._events.append(
                {
                    "event": "token",
                    "response_id": "response-1",
                    "generation_epoch": 1,
                    "payload": {"text": command["text"]},
                }
            )
        if command.get("type") == "resume_response":
            self._events.append(
                {
                    "event": "RESUME_RESPONSE",
                    "response_id": "response-1",
                    "generation_epoch": 3,
                    "payload": {
                        "response_id": "response-1",
                        "generation_epoch": 3,
                        "playback_attempt_id": 7,
                    },
                }
            )

    async def events(self):
        while self._events:
            yield self._events.pop(0)

    async def close(self):
        self.closed = True


class SlowConsumerRuntime(RecordingRealtimeRuntime):
    async def accept_command(self, command):
        self.commands.append(command)
        self._events.extend(
            [
                {"event": "token", "payload": {"text": "first"}},
                {"event": "token", "payload": {"text": "second"}},
            ]
        )

    async def events(self):
        if self._events:
            yield self._events.pop(0)
        raise SlowConsumerError("slow consumer")


def _fake_fastapi_modules():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeFastAPI
    fastapi.WebSocket = _FakeWebSocket
    fastapi.WebSocketDisconnect = _FakeWebSocketDisconnect

    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = _FileResponse
    responses.JSONResponse = _JSONResponse
    responses.Response = _Response
    return {"fastapi": fastapi, "fastapi.responses": responses}


class RealtimeWebSocketFlowTests(unittest.TestCase):
    def _create_app(self, runtime):
        with patch.dict(sys.modules, _fake_fastapi_modules()):
            return build_realtime_app(
                RealtimeServerSettings(static_dir=Path("frontend")),
                runtime_factory=lambda _session_id: runtime,
            )

    def test_unknown_session_is_accepted_then_rejected_in_band(self):
        runtime = RecordingRealtimeRuntime()
        app = self._create_app(runtime)
        websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
        websocket = _FakeWebSocket("/ws/unknown-session")

        asyncio.run(websocket_endpoint(websocket, "unknown-session"))

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent[0]["event"], "error")
        self.assertIn("unknown session", websocket.sent[0]["payload"]["message"])

    def test_query_string_session_alias_remains_backward_compatible(self):
        runtime = RecordingRealtimeRuntime()
        app = self._create_app(runtime)
        create_session = app.http_endpoints[("POST", "/sessions")]
        websocket_endpoint = app.websocket_endpoints["/ws"]
        session_id = asyncio.run(create_session()).body["session_id"]
        websocket = _FakeWebSocket(
            "/ws",
            query_params={"session_id": session_id},
            messages=[{"text": '{"type":"text","text":"hello"}'}],
        )

        asyncio.run(websocket_endpoint(websocket))

        self.assertEqual(runtime.commands[0]["type"], "text")
        self.assertEqual(websocket.sent[0]["event"], "token")

    def test_malformed_binary_frame_emits_error_without_killing_server(self):
        runtime = RecordingRealtimeRuntime()
        app = self._create_app(runtime)
        create_session = app.http_endpoints[("POST", "/sessions")]
        websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
        session_id = asyncio.run(create_session()).body["session_id"]
        websocket = _FakeWebSocket(
            f"/ws/{session_id}",
            messages=[
                {"bytes": b"bad-frame"},
                {
                    "bytes": encode_audio_frame(
                        AudioFrameHeader(sequence=1, capture_timestamp=0.02),
                        b"\x00\x00" * 320,
                    )
                },
            ],
        )

        asyncio.run(websocket_endpoint(websocket, session_id))

        self.assertEqual(websocket.sent[0]["event"], "error")
        self.assertEqual(websocket.sent[1]["event"], "AUDIO_FRAME_ACCEPTED")
        self.assertEqual(runtime.audio_sequences, [1])

    def test_disconnect_runs_runtime_disconnect_cleanup(self):
        runtime = RecordingRealtimeRuntime()
        app = self._create_app(runtime)
        create_session = app.http_endpoints[("POST", "/sessions")]
        websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
        session_id = asyncio.run(create_session()).body["session_id"]
        websocket = _FakeWebSocket(f"/ws/{session_id}")

        asyncio.run(websocket_endpoint(websocket, session_id))

        self.assertEqual(runtime.connected, 1)
        self.assertEqual(runtime.disconnected, 1)

    def test_playback_ack_and_resume_command_route_through_runtime(self):
        runtime = RecordingRealtimeRuntime()
        app = self._create_app(runtime)
        create_session = app.http_endpoints[("POST", "/sessions")]
        websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
        session_id = asyncio.run(create_session()).body["session_id"]
        websocket = _FakeWebSocket(
            f"/ws/{session_id}",
            messages=[
                {
                    "text": (
                        '{"type":"playback_ack","payload":{"response_id":"response-1",'
                        '"generation_epoch":3,"segment_id":2,"sample_offset":8,'
                        '"audio_time":0.5,"playback_attempt_id":7}}'
                    )
                },
                {"text": '{"type":"resume_response","payload":{"response_id":"response-1"}}'},
            ],
        )

        asyncio.run(websocket_endpoint(websocket, session_id))

        self.assertEqual(runtime.commands[0]["type"], "playback_ack")
        self.assertEqual(runtime.commands[1]["type"], "resume_response")
        self.assertEqual(websocket.sent[0]["event"], "RESUME_RESPONSE")
        self.assertEqual(websocket.sent[0]["payload"]["playback_attempt_id"], 7)

    def test_slow_consumer_shutdown_closes_websocket_and_runtime_session(self):
        runtime = SlowConsumerRuntime()
        app = self._create_app(runtime)
        create_session = app.http_endpoints[("POST", "/sessions")]
        websocket_endpoint = app.websocket_endpoints["/ws/{session_id}"]
        session_id = asyncio.run(create_session()).body["session_id"]
        websocket = _FakeWebSocket(
            f"/ws/{session_id}",
            messages=[{"text": '{"type":"text","text":"hello"}'}],
        )

        asyncio.run(websocket_endpoint(websocket, session_id))

        self.assertEqual(runtime.connected, 1)
        self.assertEqual(runtime.disconnected, 1)
        self.assertTrue(runtime.closed)
        self.assertIsNotNone(websocket.closed)
        self.assertEqual(websocket.closed["reason"], "slow consumer")
        self.assertNotIn(session_id, app.state.sessions)


if __name__ == "__main__":
    unittest.main()
