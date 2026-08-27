import unittest
from concurrent.futures import ThreadPoolExecutor

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

    def test_pending_reservations_are_atomic_and_cannot_overcommit(self):
        """Catches concurrent planned loads both passing against the same headroom."""
        manager = ModelManager(total_vram_bytes=100)

        def reserve(name):
            try:
                manager.reserve(name, requested_bytes=60)
                return "reserved"
            except ModelMemoryBudgetError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(reserve, ("first", "second")))

        self.assertEqual(sorted(outcomes), ["rejected", "reserved"])
        self.assertEqual(manager.diagnostics()["reserved_bytes"], 60)

    def test_release_returns_loaded_and_pending_capacity(self):
        """Catches closed or failed resources permanently leaking budget."""
        manager = ModelManager(total_vram_bytes=100)
        manager.reserve("first", requested_bytes=60)

        self.assertTrue(manager.release("first"))
        manager.reserve("second", requested_bytes=60)

        self.assertEqual(manager.diagnostics()["reserved_bytes"], 60)
        self.assertFalse(manager.release("missing"))

    def test_record_loaded_uses_measured_cuda_delta(self):
        """Catches production accounting trusting profile estimates after load."""
        snapshots = iter((10, 45))
        manager = ModelManager(
            total_vram_bytes=100,
            measure_allocated_bytes=lambda: next(snapshots),
        )
        manager.reserve("qwen", requested_bytes=20)

        record = manager.record_loaded("qwen")

        self.assertEqual(record["used_bytes"], 35)
        self.assertEqual(manager.diagnostics()["loaded_bytes"], 35)
        self.assertEqual(manager.diagnostics()["reserved_bytes"], 0)

    def test_loaded_capacity_retains_requested_floor_and_runtime_overhead(self):
        """Catches a successful load dropping its conservative floor or runtime overhead."""
        snapshots = iter((10, 22, 22))
        manager = ModelManager(
            total_vram_bytes=100,
            measure_allocated_bytes=lambda: next(snapshots),
        )
        manager.reserve(
            "qwen",
            requested_bytes=20,
            overhead_bytes={"kv_cache": 7, "cuda_context": 3},
        )

        record = manager.record_loaded("qwen", used_bytes=18)

        self.assertEqual(record["used_bytes"], 30)
        self.assertEqual(manager.diagnostics()["loaded_bytes"], 30)
        with self.assertRaises(ModelMemoryBudgetError):
            manager.reserve("cosyvoice", requested_bytes=71)

    def test_zero_measurement_never_replaces_nonzero_conservative_estimate(self):
        """Catches an unobservable subprocess load being accounted as zero bytes."""
        snapshots = iter((40, 40))
        manager = ModelManager(
            total_vram_bytes=100,
            measure_allocated_bytes=lambda: next(snapshots),
        )
        manager.reserve(
            "cosyvoice",
            requested_bytes=20,
            overhead_bytes={"worker_runtime": 5},
        )

        record = manager.record_loaded("cosyvoice", used_bytes=24)

        self.assertEqual(record["used_bytes"], 29)
        self.assertEqual(manager.diagnostics()["loaded_bytes"], 29)

    def test_measured_overage_is_rejected_after_load(self):
        """Catches a model whose real CUDA delta exceeds the safe budget."""
        snapshots = iter((10, 95))
        manager = ModelManager(
            total_vram_bytes=80,
            reserve_bytes=5,
            measure_allocated_bytes=lambda: next(snapshots),
        )
        manager.reserve("qwen", requested_bytes=20)

        with self.assertRaises(ModelMemoryBudgetError):
            manager.record_loaded("qwen")

        self.assertEqual(manager.diagnostics()["loaded_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
