import unittest

from src.core.events.events import UserInterruptEvent
from src.runtime.event_bus import EventBus
from src.runtime.lifecycle import RuntimeLifecycle
from src.runtime.pipeline import Pipeline
from src.runtime.worker import Worker


class RecordingWorker(Worker):
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    async def start(self):
        self.calls.append(f"start:{self.name}")

    async def stop(self):
        self.calls.append(f"stop:{self.name}")


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_event_bus_publish_reaches_subscribers(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe(UserInterruptEvent, handler)
        event = UserInterruptEvent("event-1", 1.0, "test", {"text": "stop"})

        await bus.publish(event)

        self.assertEqual(received, [event])

    async def test_event_bus_supports_multiple_subscribers_and_unsubscribe(self):
        bus = EventBus()
        received = []

        async def first(event):
            received.append("first")

        async def second(event):
            received.append("second")

        await bus.subscribe("USER_INTERRUPT", first)
        await bus.subscribe("USER_INTERRUPT", second)
        event = UserInterruptEvent("event-2", 2.0, "test", {})

        await bus.publish(event)
        await bus.unsubscribe("USER_INTERRUPT", first)
        await bus.publish(event)

        self.assertEqual(received, ["first", "second", "second"])

    async def test_worker_lifecycle_and_event_hook_are_async_overridable(self):
        calls = []

        class TestWorker(Worker):
            async def start(self):
                calls.append("start")

            async def stop(self):
                calls.append("stop")

            async def handle_event(self, event):
                calls.append(event.event)

        worker = TestWorker()
        event = UserInterruptEvent("event-3", 3.0, "test", {})

        await worker.start()
        await worker.handle_event(event)
        await worker.stop()

        self.assertEqual(calls, ["start", "USER_INTERRUPT", "stop"])

    async def test_pipeline_registers_and_starts_stops_workers_in_order(self):
        calls = []
        pipeline = Pipeline()
        first = RecordingWorker("first", calls)
        second = RecordingWorker("second", calls)

        pipeline.register_worker(first)
        pipeline.register_worker(second)
        await pipeline.start()
        await pipeline.stop()

        self.assertEqual(pipeline.workers, (first, second))
        self.assertEqual(
            calls,
            ["start:first", "start:second", "stop:second", "stop:first"],
        )

    async def test_runtime_lifecycle_supports_initialization_start_stop_shutdown(self):
        calls = []

        class TestLifecycle(RuntimeLifecycle):
            async def initialize(self):
                calls.append("initialize")

            async def start(self):
                calls.append("start")

            async def stop(self):
                calls.append("stop")

            async def shutdown(self):
                calls.append("shutdown")

        lifecycle = TestLifecycle()
        await lifecycle.initialize()
        await lifecycle.start()
        await lifecycle.stop()
        await lifecycle.shutdown()

        self.assertEqual(calls, ["initialize", "start", "stop", "shutdown"])
