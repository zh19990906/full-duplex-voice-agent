import unittest

from src.realtime.supervisor import WorkerSupervisor, WorkerTerminalEvent


class AlwaysFailingWorker:
    def __init__(self):
        self.start_count = 0

    async def start(self):
        self.start_count += 1
        raise RuntimeError("worker exploded")


class WorkerSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_restarts_once_then_surfaces_terminal_failure(self):
        """Catches unbounded restarts or swallowed terminal worker failure."""
        worker = AlwaysFailingWorker()
        supervisor = WorkerSupervisor("cosyvoice", max_restarts=1)

        event = await supervisor.run(worker.start)

        self.assertEqual(worker.start_count, 2)
        self.assertEqual(supervisor.status, "FAILED")
        self.assertEqual(supervisor.restart_count, 1)
        self.assertIsInstance(event, WorkerTerminalEvent)
        self.assertEqual(event.worker, "cosyvoice")
        self.assertIn("worker exploded", event.message)


if __name__ == "__main__":
    unittest.main()
