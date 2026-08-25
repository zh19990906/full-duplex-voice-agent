import argparse
import sys
import types
import typing
import unittest
from unittest.mock import patch

from scripts import run_real_server


class _FakeFastAPI:
    def __init__(self, **_kwargs):
        self.websocket_endpoints = []

    def _route(self, websocket=False):
        def decorator(endpoint):
            if websocket:
                self.websocket_endpoints.append(endpoint)
            return endpoint

        return decorator

    def get(self, _path):
        return self._route()

    def post(self, _path):
        return self._route()

    def delete(self, _path):
        return self._route()

    def websocket(self, _path):
        return self._route(websocket=True)


class RealServerAnnotationTests(unittest.TestCase):
    def test_websocket_endpoint_annotations_resolve_for_framework_injection(self):
        class WebSocket:
            pass

        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI = _FakeFastAPI
        fastapi.WebSocket = WebSocket
        fastapi.WebSocketDisconnect = RuntimeError

        responses = types.ModuleType("fastapi.responses")
        responses.FileResponse = responses.JSONResponse = responses.Response = object

        qwen = types.ModuleType("src.adapters.llm.providers.qwen_transformers")
        qwen.TransformersQwenProvider = lambda *_args, **_kwargs: object()

        cosyvoice = types.ModuleType("src.adapters.tts.providers.cosyvoice_worker")
        cosyvoice.CosyVoiceWorkerClient = lambda *_args, **_kwargs: object()

        events = types.ModuleType("src.api.events")
        events.ApiEventSerializer = object

        session = types.ModuleType("src.runtime_app.real_session")
        session.RealModelSession = object

        modules = {
            "fastapi": fastapi,
            "fastapi.responses": responses,
            "src.adapters.llm.providers.qwen_transformers": qwen,
            "src.adapters.tts.providers.cosyvoice_worker": cosyvoice,
            "src.api.events": events,
            "src.runtime_app.real_session": session,
        }
        args = argparse.Namespace(
            llm_model="llm",
            tts_model="tts",
            worker_python="python",
            worker_script="worker.py",
            cosy_root="cosy",
            prompt_audio="prompt.wav",
            prompt_text="prompt",
            worker_startup_timeout=1,
            port=8001,
        )

        with patch.dict(sys.modules, modules):
            app = run_real_server.build_app(args)

        endpoint = next(
            item for item in app.websocket_endpoints
            if item.__name__ == "websocket_session_endpoint"
        )
        hints = typing.get_type_hints(endpoint)
        self.assertIs(hints["websocket"], WebSocket)


if __name__ == "__main__":
    unittest.main()
