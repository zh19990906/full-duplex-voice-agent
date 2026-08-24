import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CosyVoiceWorkerScriptTests(unittest.TestCase):
    def test_help_does_not_import_cosyvoice_dependencies(self):
        result = subprocess.run(
            [sys.executable, "scripts/cosyvoice_worker.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)


if __name__ == "__main__":
    unittest.main()
