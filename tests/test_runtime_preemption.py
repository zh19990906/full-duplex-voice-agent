import asyncio
import unittest

from src.realtime.cancellation import ActiveTaskSlot, CancellationToken
from src.runtime.priority import Priority
from src.runtime.scheduler import RealtimeScheduler


class RuntimePreemptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_task_slot_preempts_running_task_and_awaits_cleanup(self):
        """Catches replacement that installs new work before prior cancellation cleanup."""
        slot = ActiveTaskSlot()
        stopped = asyncio.Event()

        async def old_work():
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        await slot.replace(old_work())
        await slot.replace(asyncio.sleep(0))

        await asyncio.wait_for(stopped.wait(), 0.1)
        self.assertIsNotNone(slot.task)

    async def test_scheduler_orders_the_six_realtime_priority_tiers(self):
        """Catches a scheduler whose priority values violate realtime urgency ordering."""
        scheduler = RealtimeScheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        execution_order = []

        async def block_worker():
            started.set()
            await release.wait()

        async def record(name):
            execution_order.append(name)

        await scheduler.submit(block_worker, Priority.PLAYBACK_STOP_DUCK)
        await started.wait()
        await scheduler.submit(lambda: record("background"), Priority.LATER_TTS_BACKGROUND)
        await scheduler.submit(lambda: record("main_llm"), Priority.MAIN_LLM_GENERATION)
        await scheduler.submit(lambda: record("first_tts"), Priority.FIRST_TTS_SEGMENT)
        await scheduler.submit(lambda: record("policy"), Priority.SEMANTIC_POLICY)
        await scheduler.submit(lambda: record("speech_x2"), Priority.SPEECH_START_AND_X2)
        await scheduler.submit(lambda: record("stop_duck"), Priority.PLAYBACK_STOP_DUCK)

        release.set()
        await scheduler.shutdown()

        self.assertEqual(
            execution_order,
            ["stop_duck", "speech_x2", "policy", "first_tts", "main_llm", "background"],
        )

    async def test_scheduler_skips_expired_queued_work_before_start(self):
        """Catches queued work beginning after its monotonic deadline has expired."""
        scheduler = RealtimeScheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        executed = []

        async def block_worker():
            started.set()
            await release.wait()

        async def expired_work():
            executed.append("expired")

        await scheduler.submit(block_worker, Priority.PLAYBACK_STOP_DUCK)
        await started.wait()
        deadline = asyncio.get_running_loop().time() - 1
        await scheduler.submit(expired_work, Priority.BACKGROUND, deadline=deadline)

        release.set()
        await scheduler.shutdown()

        self.assertEqual(executed, [])

    async def test_scheduler_skips_cancelled_queued_work_before_start(self):
        """Catches a cancellation token ignored while work waits in the scheduler."""
        scheduler = RealtimeScheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        executed = []
        token = CancellationToken()

        async def block_worker():
            started.set()
            await release.wait()

        async def cancelled_work():
            executed.append("cancelled")

        await scheduler.submit(block_worker, Priority.PLAYBACK_STOP_DUCK)
        await started.wait()
        await scheduler.submit(cancelled_work, Priority.BACKGROUND, cancellation_token=token)
        token.cancel()

        release.set()
        await scheduler.shutdown()

        self.assertTrue(token.is_cancelled())
        self.assertEqual(executed, [])

    def test_legacy_priority_names_alias_matching_realtime_tiers(self):
        """Catches removal of established scheduler priority imports."""
        self.assertIs(Priority.INTERRUPT, Priority.PLAYBACK_STOP_DUCK)
        self.assertIs(Priority.AUDIO, Priority.SPEECH_START_AND_X2)
        self.assertIs(Priority.TURN_EVENT, Priority.SEMANTIC_POLICY)
        self.assertIs(Priority.GENERATION, Priority.MAIN_LLM_GENERATION)
        self.assertIs(Priority.BACKGROUND, Priority.LATER_TTS_BACKGROUND)


if __name__ == "__main__":
    unittest.main()
