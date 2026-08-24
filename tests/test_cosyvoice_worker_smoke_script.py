import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CosyVoiceWorkerSmokeScriptTests(unittest.TestCase):
    def test_help_lists_required_real_model_arguments(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_cosyvoice_worker.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prompt-audio", result.stdout)
        self.assertIn("--prompt-text", result.stdout)


if __name__ == "__main__":
    unittest.main()
