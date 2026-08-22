import unittest

from src.generation.manager import GenerationManager
from src.generation.session import GenerationStatus


class GenerationManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_creates_running_active_session(self):
        manager = GenerationManager()

        session = await manager.start_generation("generation-1")

        self.assertEqual(session.id, "generation-1")
        self.assertEqual(session.status, GenerationStatus.RUNNING)
        self.assertIs(manager.active_session, session)
        self.assertIsNotNone(session.started_at)

    async def test_cancel_current_marks_session_cancelled(self):
        manager = GenerationManager()
        session = await manager.start_generation("generation-2")

        cancelled = await manager.cancel_current()

        self.assertIs(cancelled, session)
        self.assertEqual(session.status, GenerationStatus.CANCELLED)
        self.assertIsNone(manager.active_session)
        self.assertIsNotNone(session.cancelled_at)

    async def test_complete_marks_session_completed(self):
        manager = GenerationManager()
        session = await manager.start_generation("generation-3")

        completed = await manager.complete(session.id)

        self.assertIs(completed, session)
        self.assertEqual(session.status, GenerationStatus.COMPLETED)
        self.assertIsNone(manager.active_session)
        self.assertIsNotNone(session.completed_at)

    async def test_new_generation_replaces_and_cancels_old_active_session(self):
        manager = GenerationManager()
        old_session = await manager.start_generation("generation-4-old")

        new_session = await manager.start_generation("generation-4-new")

        self.assertEqual(old_session.status, GenerationStatus.CANCELLED)
        self.assertIs(manager.active_session, new_session)
        self.assertEqual(new_session.status, GenerationStatus.RUNNING)

    async def test_cancel_hook_is_notified_and_unrelated_session_is_unchanged(self):
        manager = GenerationManager()
        notifications = []

        async def on_cancel(session):
            notifications.append(session.id)

        manager.add_cancel_hook(on_cancel)
        first = await manager.start_generation("generation-5-first")
        await manager.complete(first.id)
        second = await manager.start_generation("generation-5-second")

        cancelled = await manager.cancel_current()

        self.assertIs(cancelled, second)
        self.assertEqual(notifications, [second.id])
        self.assertEqual(first.status, GenerationStatus.COMPLETED)

    async def test_cancel_current_without_active_session_is_noop(self):
        manager = GenerationManager()

        self.assertIsNone(await manager.cancel_current())
