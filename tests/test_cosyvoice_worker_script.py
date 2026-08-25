import subprocess
import sys
import unittest
import importlib.util
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

    def test_dispatch_routes_request_scoped_cancel_without_constructing_model(self):
        spec = importlib.util.spec_from_file_location(
            "cosyvoice_worker_script", ROOT / "scripts" / "cosyvoice_worker.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        worker = object.__new__(module.CosyVoiceWorker)
        calls = []
        worker._start_synthesis = lambda request: calls.append(("synthesize", request))
        worker._cancel_request = lambda request_id: calls.append(("cancel", request_id))
        worker._shutdown = lambda: calls.append(("shutdown", None))
        worker._write = lambda message: calls.append(("write", message))

        worker._dispatch({"op": "cancel", "request_id": "request-9"})
        worker._dispatch({"op": "shutdown"})

        self.assertEqual(calls, [("cancel", "request-9"), ("shutdown", None)])


if __name__ == "__main__":
    unittest.main()
