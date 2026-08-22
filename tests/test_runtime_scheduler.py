import asyncio
import unittest

from src.runtime.priority import Priority
from src.runtime.scheduler import RealtimeScheduler


class RuntimeSchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def test_priority_ordering(self):
        scheduler = RealtimeScheduler()
        execution_order = []

        async def record(name):
            execution_order.append(name)

        await scheduler.submit(lambda: record("background"), Priority.BACKGROUND)
        await scheduler.submit(lambda: record("interrupt"), Priority.INTERRUPT)
        await scheduler.submit(lambda: record("generation"), Priority.GENERATION)

        await scheduler.shutdown()

        self.assertEqual(execution_order, ["interrupt", "generation", "background"])

    async def test_async_task_executes(self):
        scheduler = RealtimeScheduler()
        completed = asyncio.Event()

        async def task():
            await asyncio.sleep(0)
            completed.set()

        await scheduler.submit(task, Priority.AUDIO)
        await scheduler.shutdown()

        self.assertTrue(completed.is_set())

    async def test_shutdown_rejects_new_tasks(self):
        scheduler = RealtimeScheduler()

        await scheduler.shutdown()

        with self.assertRaises(RuntimeError):
            await scheduler.submit(lambda: asyncio.sleep(0), Priority.BACKGROUND)

    async def test_empty_scheduler_can_shutdown(self):
        scheduler = RealtimeScheduler()

        await scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()
