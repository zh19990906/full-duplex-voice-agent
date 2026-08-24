import json
import tempfile
import unittest
from pathlib import Path

from src.agent.loop import AgentLoop
from src.application.voice_agent import VoiceAgent
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.session.agent import SessionAgent
from src.session.manager import SessionManager
from src.session.models import Session, SessionStatus
from src.session.storage import SQLiteSessionStore
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


class PersistentSessionStorageTests(unittest.TestCase):
    def test_sqlite_store_save_load_list_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.db"
            store = SQLiteSessionStore(path)
            session = Session("s1", created_at=1.5, metadata={"tenant": "test"})

            store.save_session(session)
            loaded = store.load_session("s1")

            self.assertEqual(loaded.session_id, "s1")
            self.assertEqual(loaded.created_at, 1.5)
            self.assertEqual(loaded.metadata["tenant"], "test")
            self.assertEqual([item.session_id for item in store.list_sessions()], ["s1"])
            store.delete_session("s1")
            with self.assertRaises(KeyError):
                store.load_session("s1")
            store.close()

    def test_session_and_memory_restore_after_manager_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.db"
            first_store = SQLiteSessionStore(path)
            first = SessionManager(session_store=first_store, agent_factory=make_agent)
            session = first.create_session("s1", {"tenant": "test"})
            session.memory_manager.add_message("s1", "user", "remember this", timestamp=1.0)
            state = session.memory_manager.get_session("s1")
            state.summary = "summary"
            state.current_state["step"] = 2
            session.memory_manager.save_state(state)
            session.agent.agent_loop.state.iteration = 3
            first.close_session("s1")
            first_store.close()

            second_store = SQLiteSessionStore(path)
            second = SessionManager(session_store=second_store, agent_factory=make_agent)
            restored = second.get_session("s1")

            self.assertEqual(restored.status, SessionStatus.CLOSED)
            self.assertEqual(restored.metadata["tenant"], "test")
            self.assertEqual([m.content for m in restored.memory_manager.get_context("s1")], ["remember this"])
            self.assertEqual(restored.memory_manager.get_session("s1").summary, "summary")
            self.assertEqual(restored.memory_manager.get_session("s1").current_state["step"], 2)
            self.assertEqual(restored.agent.agent_loop.state.iteration, 3)
            second_store.close()

    def test_delete_removes_persisted_session_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.db")
            manager = SessionManager(session_store=store)
            session = manager.create_session("s1")
            session.memory_manager.add_message("s1", "user", "temporary")
            manager.close_session("s1")
            manager.delete_session("s1")

            self.assertEqual(store.list_sessions(), ())
            with self.assertRaises(KeyError):
                store.get_messages("s1")
            store.close()

    def test_in_memory_mode_remains_compatible(self):
        manager = SessionManager()
        session = manager.create_session("s1")
        session.memory_manager.add_message("s1", "user", "local")

        self.assertEqual([m.content for m in session.memory_manager.get_context("s1")], ["local"])
        self.assertEqual(manager.get_session("s1").session_id, "s1")

    def test_multiple_persisted_sessions_remain_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.db")
            manager = SessionManager(session_store=store)
            first = manager.create_session("a")
            second = manager.create_session("b")
            first.memory_manager.add_message("a", "user", "A")
            second.memory_manager.add_message("b", "user", "B")
            manager.close_session("a")
            manager.close_session("b")
            store.close()

            restored_store = SQLiteSessionStore(Path(directory) / "sessions.db")
            restored = SessionManager(session_store=restored_store)

            self.assertEqual([m.content for m in restored.get_session("a").memory_manager.get_context("a")], ["A"])
            self.assertEqual([m.content for m in restored.get_session("b").memory_manager.get_context("b")], ["B"])
            restored_store.close()


if __name__ == "__main__":
    unittest.main()
