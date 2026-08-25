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

    async def test_active_task_slot_serializes_concurrent_replacements_without_orphaning_work(self):
        """Catches concurrent replacements returning one task while orphaning the other."""
        slot = ActiveTaskSlot()
        old_cleanup_started = asyncio.Event()
        allow_old_cleanup = asyncio.Event()
        first_stopped = asyncio.Event()
        second_stopped = asyncio.Event()

        async def old_work():
            try:
                await asyncio.Future()
            finally:
                old_cleanup_started.set()
                await allow_old_cleanup.wait()

        async def first_work():
            try:
                await asyncio.Future()
            finally:
                first_stopped.set()

        async def second_work():
            try:
                await asyncio.Future()
            finally:
                second_stopped.set()

        await slot.replace(old_work())
        first_replacement = asyncio.create_task(slot.replace(first_work()))
        await old_cleanup_started.wait()
        second_replacement = asyncio.create_task(slot.replace(second_work()))
        allow_old_cleanup.set()

        first_task, second_task = await asyncio.gather(
            first_replacement,
            second_replacement,
        )

        self.assertIsNot(first_task, second_task)
        self.assertIs(slot.task, second_task)
        self.assertTrue(first_stopped.is_set())

        await slot.cancel()
        self.assertTrue(second_stopped.is_set())

    async def test_active_task_slot_records_old_failure_and_installs_replacement(self):
        """Catches old-task failures that leak supplied replacement coroutines."""
        slot = ActiveTaskSlot()
        old_failed = asyncio.Event()
        replacement_stopped = asyncio.Event()

        async def failed_work():
            old_failed.set()
            raise RuntimeError("old work failed")

        async def replacement_work():
            try:
                await asyncio.Future()
            finally:
                replacement_stopped.set()

        await slot.replace(failed_work())
        await old_failed.wait()
        replacement_coroutine = replacement_work()
        try:
            replacement_task = await slot.replace(replacement_coroutine)

            self.assertIs(slot.task, replacement_task)
            self.assertIsInstance(slot.last_error, RuntimeError)
            await slot.cancel()
            self.assertIsNone(slot.task)
            self.assertTrue(replacement_stopped.is_set())
            self.assertIsNone(replacement_coroutine.cr_frame)
        finally:
            if replacement_coroutine.cr_frame is not None:
                replacement_coroutine.close()

    async def test_active_task_slot_cancel_records_failure_and_clears_slot(self):
        """Catches cancellation that propagates failure and leaves a stale task slot."""
        slot = ActiveTaskSlot()
        failed = asyncio.Event()

        async def failed_work():
            failed.set()
            raise RuntimeError("work failed")

        await slot.replace(failed_work())
        await failed.wait()
        await slot.cancel()

        self.assertIsNone(slot.task)
        self.assertIsInstance(slot.last_error, RuntimeError)

    async def test_cancelled_replace_waits_for_old_cleanup_before_releasing_slot(self):
        """Catches a cancelled replace caller releasing ownership during old cleanup."""
        slot = ActiveTaskSlot()
        old_cleanup_started = asyncio.Event()
        allow_old_cleanup = asyncio.Event()
        uninstalled_started = asyncio.Event()
        later_started = asyncio.Event()
        later_stopped = asyncio.Event()

        async def old_work():
            try:
                await asyncio.Future()
            finally:
                old_cleanup_started.set()
                await allow_old_cleanup.wait()

        async def uninstalled_work():
            uninstalled_started.set()
            await asyncio.Future()

        async def later_work():
            try:
                later_started.set()
                await asyncio.Future()
            finally:
                later_stopped.set()

        await slot.replace(old_work())
        uninstalled_coroutine = uninstalled_work()
        cancelled_replace = asyncio.create_task(slot.replace(uninstalled_coroutine))
        await old_cleanup_started.wait()
        cancelled_replace.cancel()
        await asyncio.sleep(0)
        replace_waits_for_cleanup = not cancelled_replace.done()

        later_replace = asyncio.create_task(slot.replace(later_work()))
        await asyncio.sleep(0)
        later_waits_for_cleanup = not later_started.is_set()
        allow_old_cleanup.set()

        with self.assertRaises(asyncio.CancelledError):
            await cancelled_replace
        later_task = await later_replace
        self.assertIs(slot.task, later_task)
        self.assertTrue(replace_waits_for_cleanup)
        self.assertTrue(later_waits_for_cleanup)
        self.assertFalse(uninstalled_started.is_set())
        self.assertIsNone(uninstalled_coroutine.cr_frame)

        await slot.cancel()
        self.assertTrue(later_stopped.is_set())

    async def test_cancelled_cancel_waits_for_old_cleanup_before_releasing_slot(self):
        """Catches a cancelled cancel caller releasing ownership during old cleanup."""
        slot = ActiveTaskSlot()
        old_cleanup_started = asyncio.Event()
        allow_old_cleanup = asyncio.Event()
        later_started = asyncio.Event()
        later_stopped = asyncio.Event()

        async def old_work():
            try:
                await asyncio.Future()
            finally:
                old_cleanup_started.set()
                await allow_old_cleanup.wait()

        async def later_work():
            try:
                later_started.set()
                await asyncio.Future()
            finally:
                later_stopped.set()

        await slot.replace(old_work())
        cancelled_cancel = asyncio.create_task(slot.cancel())
        await old_cleanup_started.wait()
        cancelled_cancel.cancel()
        await asyncio.sleep(0)
        cancel_waits_for_cleanup = not cancelled_cancel.done()

        later_replace = asyncio.create_task(slot.replace(later_work()))
        await asyncio.sleep(0)
        later_waits_for_cleanup = not later_started.is_set()
        allow_old_cleanup.set()

        with self.assertRaises(asyncio.CancelledError):
            await cancelled_cancel
        later_task = await later_replace
        self.assertIs(slot.task, later_task)
        self.assertTrue(cancel_waits_for_cleanup)
        self.assertTrue(later_waits_for_cleanup)

        await slot.cancel()
        self.assertTrue(later_stopped.is_set())

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
        self.assertIs(Priority.TURN_EVENT, Priority.SPEECH_START_AND_X2)
        self.assertIs(Priority.GENERATION, Priority.MAIN_LLM_GENERATION)
        self.assertIs(Priority.BACKGROUND, Priority.LATER_TTS_BACKGROUND)


if __name__ == "__main__":
    unittest.main()
