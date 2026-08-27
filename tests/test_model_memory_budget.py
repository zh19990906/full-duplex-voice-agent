import unittest

from src.model_runtime.manager import ModelManager, ModelMemoryBudgetError


GIB = 1024**3


class ModelMemoryBudgetTests(unittest.TestCase):
    def test_model_manager_rejects_load_above_vram_budget(self):
        """Catches model admission that exceeds the configured safe GPU budget."""
        manager = ModelManager(total_vram_bytes=72 * GIB, reserve_bytes=10 * GIB)
        manager.record_loaded("qwen", used_bytes=58 * GIB)

        with self.assertRaises(ModelMemoryBudgetError):
            manager.reserve("cosyvoice", requested_bytes=8 * GIB)

    def test_model_manager_records_before_after_and_overhead_diagnostics(self):
        """Catches budget reports that omit the measurements operators need to debug OOM risk."""
        snapshots = iter((24 * GIB, 38 * GIB))
        manager = ModelManager(
            total_vram_bytes=72 * GIB,
            reserve_bytes=10 * GIB,
            measure_allocated_bytes=lambda: next(snapshots),
        )

        manager.reserve(
            "qwen",
            requested_bytes=12 * GIB,
            overhead_bytes={
                "kv_cache": 4 * GIB,
                "cuda_context": 2 * GIB,
            },
        )
        manager.record_loaded("qwen", used_bytes=14 * GIB)
        diagnostics = manager.diagnostics()

        self.assertEqual(diagnostics["models"]["qwen"]["before_bytes"], 24 * GIB)
        self.assertEqual(diagnostics["models"]["qwen"]["after_bytes"], 38 * GIB)
        self.assertEqual(diagnostics["models"]["qwen"]["requested_bytes"], 12 * GIB)
        self.assertEqual(diagnostics["models"]["qwen"]["overhead_bytes"]["kv_cache"], 4 * GIB)


if __name__ == "__main__":
    unittest.main()
