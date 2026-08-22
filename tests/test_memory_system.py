import json
import unittest

from src.application.voice_agent import VoiceAgent
from src.config import load_yaml
from src.core.events.events import UserTurnEndEvent
from src.memory.context import ContextBuilder
from src.memory.manager import MemoryManager
from src.memory.models import ConversationMessage, ConversationState
from src.memory.store import InMemoryStore
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.runtime.event_bus import EventBus


class MemorySystemTests(unittest.IsolatedAsyncioTestCase):
    def test_message_and_state_are_serializable(self):
        message = ConversationMessage(
            role="user",
            content="Hello",
            timestamp=1.0,
            metadata={"session_id": "s1", "tags": ["greeting"]},
        )
        state = ConversationState(
            session_id="s1",
            summary="A greeting",
            user_preferences={"language": "en"},
            current_state={"mode": "listening"},
        )

        self.assertEqual(json.loads(json.dumps(message.to_dict()))["role"], "user")
        self.assertEqual(state.to_dict()["user_preferences"]["language"], "en")

    def test_short_term_memory_preserves_order_and_limit(self):
        manager = MemoryManager(store=InMemoryStore(), max_messages=2)
        manager.create_session("s1")
        manager.add_message("s1", "user", "one", timestamp=1.0)
        manager.add_message("s1", "assistant", "two", timestamp=2.0)
        manager.add_message("s1", "user", "three", timestamp=3.0)

        messages = manager.get_context("s1")

        self.assertEqual([message.content for message in messages], ["two", "three"])

    def test_session_lifecycle_and_state_persistence(self):
        store = InMemoryStore()
        manager = MemoryManager(store=store)
        state = manager.create_session("s1")
        state.summary = "summary"
        manager.save_state(state)
        manager.add_message("s1", "user", "hello")

        reopened = MemoryManager(store=store)
        self.assertEqual([message.content for message in reopened.get_context("s1")], ["hello"])

        self.assertEqual(manager.get_session("s1").summary, "summary")
        manager.clear("s1")
        self.assertEqual(manager.get_context("s1"), ())
        self.assertEqual(manager.get_session("s1").summary, "summary")
        manager.delete_session("s1")
        with self.assertRaises(KeyError):
            manager.get_session("s1")

    def test_context_builder_returns_structured_context(self):
        manager = MemoryManager()
        manager.create_session("s1")
        manager.add_message("s1", "user", "previous")
        builder = ContextBuilder(manager)

        context = builder.build("s1", current_user_message="next")

        self.assertEqual([message.role for message in context], ["user", "user"])
        self.assertEqual(context[-1].content, "next")

    async def test_voice_agent_records_user_and_assistant_context(self):
        memory = MemoryManager()
        memory.create_session("s1")
        agent = VoiceAgent(
            EventBus(),
            ConversationController(),
            GenerationManager(),
            memory_manager=memory,
            memory_session_id="s1",
        )
        await agent.start()
        event = UserTurnEndEvent("event-1", 1.0, "asr", {"text": "Hello"})

        await agent.handle_event(event)
        agent.record_assistant_response("Hi there", timestamp=2.0)

        self.assertEqual(
            [message.content for message in memory.get_context("s1")],
            ["Hello", "Hi there"],
        )
        self.assertEqual(
            [message.content for message in agent.generation_context],
            ["Hello", "Hi there"],
        )
        await agent.stop()

    def test_memory_config_is_dependency_free(self):
        config = load_yaml("configs/memory.yaml")
        self.assertEqual(config["memory"]["provider"], "in_memory")
        self.assertEqual(config["memory"]["short_term"]["max_messages"], 20)


if __name__ == "__main__":
    unittest.main()
