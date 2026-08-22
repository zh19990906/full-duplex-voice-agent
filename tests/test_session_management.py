import asyncio
import unittest

from src.agent.loop import AgentLoop
from src.application.voice_agent import VoiceAgent
from src.config import load_yaml
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.session.agent import SessionAgent
from src.session.manager import SessionManager
from src.session.models import SessionStatus
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter


def make_agent(session, memory):
    async def decide(_context):
        return f"reply for {session.session_id}"

    loop = AgentLoop(
        decide,
        memory_manager=memory,
        tool_router=ToolRouter(ToolRegistry()),
        session_id=session.session_id,
    )
    voice_agent = VoiceAgent(
        EventBus(),
        ConversationController(),
        GenerationManager(),
        memory_manager=memory,
        memory_session_id=session.session_id,
        agent_loop=loop,
    )
    return SessionAgent(session, memory, voice_agent=voice_agent, agent_loop=loop)


class SessionManagementTests(unittest.IsolatedAsyncioTestCase):
    def test_session_lifecycle_create_activate_close_delete(self):
        manager = SessionManager()

        session = manager.create_session("s1")

        self.assertEqual(session.status, SessionStatus.CREATED)
        self.assertEqual(session.to_dict()["status"], "CREATED")
        self.assertEqual(session.to_dict()["resources"]["memory"]["owner"], "s1")
        manager.activate_session("s1")
        self.assertEqual(manager.get_session("s1").status, SessionStatus.ACTIVE)
        manager.mark_idle("s1")
        self.assertEqual(manager.get_session("s1").status, SessionStatus.IDLE)
        manager.close_session("s1")
        self.assertEqual(manager.get_session("s1").status, SessionStatus.CLOSED)
        manager.delete_session("s1")
        with self.assertRaises(KeyError):
            manager.get_session("s1")

    def test_invalid_session_transition_fails_clearly(self):
        manager = SessionManager()
        manager.create_session("s1")

        with self.assertRaises(ValueError):
            manager.mark_idle("s1")
        manager.close_session("s1")
        with self.assertRaises(ValueError):
            manager.activate_session("s1")

    def test_sessions_have_isolated_memory_and_agent_state(self):
        manager = SessionManager(agent_factory=make_agent)
        first = manager.create_session("a")
        second = manager.create_session("b")

        self.assertIsNot(first.memory_manager, second.memory_manager)
        self.assertIsNot(first.agent, second.agent)
        first.memory_manager.add_message("a", "user", "private A")
        second.memory_manager.add_message("b", "user", "private B")
        self.assertEqual([m.content for m in first.memory_manager.get_context("a")], ["private A"])
        self.assertEqual([m.content for m in second.memory_manager.get_context("b")], ["private B"])
        self.assertEqual(first.agent.agent_loop.state.session_id, "a")
        self.assertEqual(second.agent.agent_loop.state.session_id, "b")

    async def test_parallel_sessions_do_not_leak_context(self):
        manager = SessionManager(agent_factory=make_agent)
        first = manager.create_session("a")
        second = manager.create_session("b")
        manager.activate_session("a")
        manager.activate_session("b")

        results = await asyncio.gather(first.agent.run("hello A"), second.agent.run("hello B"))

        self.assertEqual(results, ["reply for a", "reply for b"])
        self.assertIn("hello A", [m.content for m in first.memory_manager.get_context("a")])
        self.assertNotIn("hello B", [m.content for m in first.memory_manager.get_context("a")])
        self.assertIn("hello B", [m.content for m in second.memory_manager.get_context("b")])
        self.assertNotIn("hello A", [m.content for m in second.memory_manager.get_context("b")])

    async def test_same_session_execution_is_serialized(self):
        manager = SessionManager(agent_factory=make_agent)
        session = manager.create_session("s1")
        manager.activate_session("s1")

        results = await asyncio.gather(session.agent.run("one"), session.agent.run("two"))

        self.assertEqual(results, ["reply for s1", "reply for s1"])
        self.assertEqual(
            [message.content for message in session.memory_manager.get_context("s1")],
            ["one", "reply for s1", "two", "reply for s1"],
        )

    def test_resources_are_owned_by_their_session(self):
        manager = SessionManager(agent_factory=make_agent)
        session = manager.create_session("s1")

        resources = session.resources

        self.assertEqual(resources["memory"].session_id, "s1")
        self.assertEqual(resources["agent"].session_id, "s1")
        self.assertEqual({resource.owner for resource in resources.values()}, {"s1"})

    def test_session_config_is_dependency_free(self):
        config = load_yaml("configs/session.yaml")

        self.assertEqual(config["session"]["max_sessions"], 100)
        self.assertEqual(config["session"]["idle_timeout_seconds"], 1800)


if __name__ == "__main__":
    unittest.main()
