import subprocess
import sys
import unittest
from pathlib import Path


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
