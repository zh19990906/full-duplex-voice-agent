import subprocess
import sys
import unittest
from pathlib import Path


class QwenSmokeScriptTests(unittest.TestCase):
    def test_script_help_works_when_invoked_by_path(self):
        repository_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/test_qwen_provider.py", "--help"],
            cwd=repository_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)


if __name__ == "__main__":
    unittest.main()
