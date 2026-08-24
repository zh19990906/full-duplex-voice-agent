import asyncio
import unittest

from src.agent.loop import AgentLoop
from src.api.events import ApiEventSerializer
from src.api.models import ApiResponse
from src.api.service import ApiService
from src.api.websocket import WebSocketEndpoint
from src.application.voice_agent import VoiceAgent
from src.controller.controller import ConversationController
from src.core.events.events import UserSpeechPartialEvent
from src.generation.manager import GenerationManager
from src.llm_runtime.stream import TokenChunk
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.session.agent import SessionAgent
from src.session.manager import SessionManager
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter
from src.tts_runtime.stream import AudioChunk


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


def make_service(audio_handler=None):
    manager = SessionManager(agent_factory=make_agent)
    return ApiService(manager), manager, audio_handler


class ApiServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_create_get_and_delete_delegate_to_manager(self):
        service, manager, _ = make_service()

        created = await service.handle("POST", "/sessions", {"session_id": "s1"})
        fetched = await service.handle("GET", "/sessions/s1")
        deleted = await service.handle("DELETE", "/sessions/s1")

        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.body["session_id"], "s1")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(deleted.status_code, 204)
        with self.assertRaises(KeyError):
            manager.get_session("s1")

    async def test_message_is_executed_by_session_agent(self):
        service, _, _ = make_service()
        await service.handle("POST", "/sessions", {"session_id": "s1"})

        response = await service.handle(
            "POST", "/sessions/s1/message", {"message": "hello"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body["response"], "reply for s1")

    async def test_errors_are_serialized_without_raising(self):
        service, _, _ = make_service()

        missing = await service.handle("GET", "/sessions/missing")
        invalid = await service.handle("POST", "/sessions/s1/message", {})
        unknown = await service.handle("PATCH", "/unknown")

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid.status_code, 404)
        self.assertEqual(unknown.status_code, 404)

    async def test_concurrent_sessions_are_isolated(self):
        service, _, _ = make_service()
        await asyncio.gather(
            service.handle("POST", "/sessions", {"session_id": "a"}),
            service.handle("POST", "/sessions", {"session_id": "b"}),
        )

        responses = await asyncio.gather(
            service.handle("POST", "/sessions/a/message", {"message": "A"}),
            service.handle("POST", "/sessions/b/message", {"message": "B"}),
        )

        self.assertEqual([response.body["response"] for response in responses], ["reply for a", "reply for b"])

    async def test_closing_one_session_does_not_affect_another(self):
        service, manager, _ = make_service()
        await asyncio.gather(
            service.handle("POST", "/sessions", {"session_id": "a"}),
            service.handle("POST", "/sessions", {"session_id": "b"}),
        )

        self.assertEqual((await service.handle("DELETE", "/sessions/a")).status_code, 204)
        self.assertEqual((await service.handle("GET", "/sessions/b")).status_code, 200)
        self.assertEqual((await service.handle("POST", "/sessions/b/message", {"message": "still here"})).status_code, 200)
        with self.assertRaises(KeyError):
            manager.get_session("a")

    async def test_websocket_event_order_and_serialization(self):
        service, manager, _ = make_service()
        await service.handle("POST", "/sessions", {"session_id": "s1"})
        endpoint = WebSocketEndpoint(manager, service.serializer)
        connection = await endpoint.connect("s1")

        await connection.publish(UserSpeechPartialEvent("e1", 1.0, "asr", {"text": "hel"}))
        await connection.publish(TokenChunk("t1", "Hello", 2.0, False))
        await connection.publish(AudioChunk("a1", b"pcm", 3.0, False))

        events = [await connection.receive_event() for _ in range(3)]

        self.assertEqual([event["event"] for event in events], ["USER_SPEECH_PARTIAL", "token", "audio_chunk"])
        self.assertEqual(events[0]["session_id"], "s1")
        self.assertEqual(events[1]["payload"]["text"], "Hello")
        self.assertEqual(events[2]["payload"]["audio_data"], "cGNt")

    async def test_audio_message_is_forwarded_without_api_decoding(self):
        received = []

        async def audio_handler(session, payload):
            received.append((session.session_id, payload))

        service, manager, _ = make_service(audio_handler)
        await service.handle("POST", "/sessions", {"session_id": "s1"})
        connection = await WebSocketEndpoint(manager, service.serializer, audio_handler=audio_handler).connect("s1")

        await connection.receive({"type": "audio", "data": b"raw-pcm"})

        self.assertEqual(received, [("s1", b"raw-pcm")])

    def test_event_serializer_keeps_core_event_contract_unchanged(self):
        event = UserSpeechPartialEvent("e1", 1.0, "asr", {"text": "hi"})

        serialized = ApiEventSerializer.serialize(event, "s1")

        self.assertEqual(event.to_dict()["event"], "USER_SPEECH_PARTIAL")
        self.assertEqual(serialized["payload"], {"text": "hi"})
        self.assertEqual(serialized["session_id"], "s1")

    def test_api_response_is_serializable(self):
        response = ApiResponse(200, {"ok": True})

        self.assertEqual(response.to_dict(), {"status_code": 200, "body": {"ok": True}})


if __name__ == "__main__":
    unittest.main()
