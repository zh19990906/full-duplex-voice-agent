import subprocess
import sys
import unittest
from pathlib import Path

from scripts import run_real_server


ROOT = Path(__file__).resolve().parents[1]


class RealServerScriptTests(unittest.TestCase):
    def test_help_defaults_to_mapped_frontend_port(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_real_server.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("8001", result.stdout)
        self.assertIn("--llm-model", result.stdout)
        self.assertIn("--tts-model", result.stdout)

    def test_parse_args_preserves_cli_overrides_for_real_adapter_paths(self):
        args = run_real_server.parse_args(
            [
                "--llm-model",
                "/models/qwen",
                "--tts-model",
                "/models/cosyvoice",
                "--asr-model",
                "/models/faster-whisper",
                "--turn-model",
                "/models/x2-turn",
                "--worker-python",
                "/envs/cosy/bin/python",
                "--cosy-root",
                "/srv/CosyVoice",
                "--prompt-audio",
                "/tmp/prompt.wav",
                "--prompt-text",
                "hello",
            ]
        )

        self.assertEqual(args.llm_model, "/models/qwen")
        self.assertEqual(args.tts_model, "/models/cosyvoice")
        self.assertEqual(args.asr_model, "/models/faster-whisper")
        self.assertEqual(args.turn_model, "/models/x2-turn")
        self.assertEqual(args.worker_python, "/envs/cosy/bin/python")
        self.assertEqual(args.cosy_root, "/srv/CosyVoice")

    def test_real_server_has_no_direct_serial_model_session_path(self):
        source = (ROOT / "scripts" / "run_real_server.py").read_text(encoding="utf-8")

        self.assertNotIn("RealModelSession", source)
        self.assertNotIn("model_lock", source)
        self.assertNotIn(".model.run(", source)


if __name__ == "__main__":
    unittest.main()
