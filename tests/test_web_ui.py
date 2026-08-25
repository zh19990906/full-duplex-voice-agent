import json
import re
import unittest
from pathlib import Path


FRONTEND = Path(__file__).parents[1] / "frontend"


class WebUiTests(unittest.TestCase):
    def test_frontend_contains_required_files(self):
        required = {
            "index.html",
            "config.js",
            "style.css",
            "README.md",
            "src/app.js",
            "src/api.js",
            "src/websocket.js",
            "src/audio.js",
            "src/timeline.js",
        }

        self.assertEqual(
            required,
            {path.relative_to(FRONTEND).as_posix() for path in FRONTEND.rglob("*") if path.is_file()},
        )

    def test_frontend_visible_copy_is_chinese(self):
        index = (FRONTEND / "index.html").read_text(encoding="utf-8")
        for label in ("全双工语音助手", "创建会话", "关闭会话", "发送", "开启麦克风", "事件时间线"):
            self.assertIn(label, index)

    def test_api_endpoints_are_configurable(self):
        config = (FRONTEND / "config.js").read_text(encoding="utf-8")
        self.assertIn("api", config)
        self.assertIn("websocket", config)
        self.assertNotRegex(config, r"https?://[^\"']+production")

        api = (FRONTEND / "src/api.js").read_text(encoding="utf-8")
        for endpoint in ("/sessions", "/message"):
            self.assertIn(endpoint, api)

    def test_websocket_event_mapping_is_explicit(self):
        websocket = (FRONTEND / "src/websocket.js").read_text(encoding="utf-8")
        for event_name in ("transcript", "token", "audio", "tool_call", "tool_result", "agent_state"):
            self.assertIn(event_name, websocket)
        self.assertIn("addEventListener", websocket)

    def test_audio_module_uses_browser_boundaries_only(self):
        audio = (FRONTEND / "src/audio.js").read_text(encoding="utf-8")
        self.assertIn("MediaRecorder", audio)
        self.assertIn("AudioContext", audio)
        self.assertIn("Int16Array", audio)
        self.assertIn("createBuffer", audio)
        self.assertIn("context.resume", audio)
        self.assertIn("async unlock", audio)

    def test_frontend_has_no_backend_imports_or_framework_dependencies(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in FRONTEND.rglob("*.js"))
        self.assertNotIn("from src", source)
        self.assertNotIn("require(", source)
        self.assertNotIn("react", source.lower())
        self.assertNotIn("vue", source.lower())


if __name__ == "__main__":
    unittest.main()
