import unittest

from src.task.models import TaskState, TaskStatus
from src.task.state_manager import TaskStateManager
from src.task.storage import InMemoryTaskStorage


class TaskStateManagerTest(unittest.TestCase):
    def setUp(self):
        self.manager = TaskStateManager(InMemoryTaskStorage())

    def test_create_task_stores_generic_state(self):
        task = self.manager.create_task(
            task_id="count-001",
            task_type="counting",
            state_data={"current_number": 5},
        )

        self.assertEqual(
            task,
            TaskState(
                task_id="count-001",
                task_type="counting",
                status=TaskStatus.RUNNING,
                state_data={"current_number": 5},
            ),
        )
        self.assertEqual(self.manager.get_task("count-001"), task)

    def test_update_state_replaces_state_data(self):
        self.manager.create_task("task-1", "generic", {"value": 1})

        updated = self.manager.update_state("task-1", {"value": 2})

        self.assertEqual(updated.state_data, {"value": 2})
        self.assertEqual(self.manager.get_task("task-1").state_data, {"value": 2})

    def test_pause_and_resume_preserve_state_data(self):
        self.manager.create_task("count-001", "counting", {"current_number": 5})

        paused = self.manager.pause_task("count-001")
        resumed = self.manager.resume_task("count-001")

        self.assertEqual(paused.status, TaskStatus.PAUSED)
        self.assertEqual(resumed.status, TaskStatus.RUNNING)
        self.assertEqual(resumed.state_data, {"current_number": 5})

    def test_storage_supports_save_load_delete(self):
        storage = InMemoryTaskStorage()
        task = TaskState("task-2", "generic", TaskStatus.RUNNING, {})

        storage.save(task)
        self.assertEqual(storage.load("task-2"), task)
        storage.delete("task-2")
        self.assertIsNone(storage.load("task-2"))

    def test_missing_task_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.manager.get_task("missing")
        with self.assertRaises(KeyError):
            self.manager.update_state("missing", {})
        with self.assertRaises(KeyError):
            self.manager.pause_task("missing")
        with self.assertRaises(KeyError):
            self.manager.resume_task("missing")
