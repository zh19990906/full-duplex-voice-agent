import subprocess
import sys
import unittest
from pathlib import Path


class ApplicationImportTests(unittest.TestCase):
    def test_application_voice_agent_imports_without_circular_dependency(self):
        repository_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import src.application.voice_agent; from src.agent import AgentLoop",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
