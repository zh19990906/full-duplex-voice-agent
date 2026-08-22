import unittest

from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus
from src.runtime.scheduler import RealtimeScheduler
from src.runtime_app.bootstrap import create_application, initialize_application
from src.runtime_app.container import ApplicationContainer


class ApplicationBootstrapTest(unittest.IsolatedAsyncioTestCase):
    async def test_application_container_creates(self):
        app = create_application()

        self.assertIsInstance(app, ApplicationContainer)
        self.assertIsInstance(app.event_bus, EventBus)
        self.assertIsInstance(app.controller, ConversationController)
        self.assertIsInstance(app.generation_manager, GenerationManager)
        self.assertIsInstance(app.scheduler, RealtimeScheduler)

    async def test_components_are_wired_together(self):
        app = create_application()

        self.assertIs(app.voice_agent.event_bus, app.event_bus)
        self.assertIs(app.voice_agent.controller, app.controller)
        self.assertIs(app.voice_agent.generation_manager, app.generation_manager)
        self.assertIs(app.voice_agent.action_executor.generation_manager, app.generation_manager)

    async def test_bootstrap_initializes_application(self):
        app = create_application()

        await app.initialize()

        self.assertTrue(app.initialized)
        self.assertTrue(app.voice_agent._started)
        await app.shutdown()
        self.assertFalse(app.initialized)

    async def test_initialize_application_helper_returns_initialized_container(self):
        app = await initialize_application()

        self.assertTrue(app.initialized)
        await app.shutdown()


if __name__ == "__main__":
    unittest.main()
