import asyncio
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

    async def test_worker_waits_for_configured_backoff_before_single_restart(self):
        """Catches hot-loop restarts that immediately hammer a failed worker."""
        attempts = 0
        delays = []

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient")
            return "ready"

        async def record_sleep(delay):
            delays.append(delay)

        supervisor = WorkerSupervisor(
            "cosyvoice",
            max_restarts=1,
            restart_backoff_seconds=0.25,
            sleep=record_sleep,
        )

        result = await supervisor.run(operation)

        self.assertEqual(result, "ready")
        self.assertEqual(attempts, 2)
        self.assertEqual(delays, [0.25])

    async def test_cancellation_during_backoff_propagates_without_terminal_failure(self):
        """Catches cancellation being swallowed and reported as a worker crash."""
        sleeping = asyncio.Event()

        async def blocked_sleep(_delay):
            sleeping.set()
            await asyncio.Event().wait()

        async def operation():
            raise RuntimeError("restart me")

        supervisor = WorkerSupervisor(
            "cosyvoice",
            restart_backoff_seconds=1.0,
            sleep=blocked_sleep,
        )
        task = asyncio.create_task(supervisor.run(operation))
        await sleeping.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(supervisor.status, "CANCELLED")
        self.assertNotEqual(supervisor.status, "FAILED")

    async def test_close_is_idempotent_and_cancels_pending_restart(self):
        """Catches close leaving a restart task alive or failing on repeated teardown."""
        sleeping = asyncio.Event()

        async def blocked_sleep(_delay):
            sleeping.set()
            await asyncio.Event().wait()

        async def operation():
            raise RuntimeError("restart me")

        supervisor = WorkerSupervisor(
            "cosyvoice",
            restart_backoff_seconds=1.0,
            sleep=blocked_sleep,
        )
        task = asyncio.create_task(supervisor.run(operation))
        await sleeping.wait()

        await supervisor.close()
        await supervisor.close()

        self.assertTrue(task.cancelled())
        self.assertEqual(supervisor.status, "CLOSED")


if __name__ == "__main__":
    unittest.main()
