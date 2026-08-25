import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class FrontendPlaybackTests(unittest.TestCase):
    def test_browser_playback_queue_behavior(self):
        if shutil.which("node") is None:
            self.skipTest("Node.js is required for the browser module behavior test")

        result = subprocess.run(
            ["node", "--test", "tests/frontend_playback_test.mjs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
