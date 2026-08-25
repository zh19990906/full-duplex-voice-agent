import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RealStackScriptTests(unittest.TestCase):
    def test_help_lists_all_real_model_inputs(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_real_stack.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for argument in ("--asr-model", "--llm-model", "--tts-model", "--audio", "--output"):
            self.assertIn(argument, result.stdout)


if __name__ == "__main__":
    unittest.main()
