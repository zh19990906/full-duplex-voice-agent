import inspect
import unittest
from dataclasses import asdict

from src.core.events.events import (
    AssistantSpeechChunkEvent,
    UserInterruptEvent,
)
from src.core.interfaces.asr import ASRAdapter
from src.core.interfaces.llm import LLMAdapter
from src.core.interfaces.translation import TranslationAdapter
from src.core.interfaces.turn import TurnAdapter
from src.core.interfaces.tts import TTSAdapter
from src.core.state.state import ConversationState, RuntimeState, TaskState


class CoreContractsTest(unittest.TestCase):
    def test_event_creation_preserves_common_fields_and_event_name(self):
        event = UserInterruptEvent(
            event_id="evt-1",
            timestamp=123.5,
            source="turn-adapter",
            payload={"transcript": "等等", "confidence": 0.95},
        )

        self.assertEqual(event.event, "USER_INTERRUPT")
        self.assertEqual(event.event_id, "evt-1")
        self.assertEqual(event.timestamp, 123.5)
        self.assertEqual(event.source, "turn-adapter")
        self.assertEqual(event.payload["transcript"], "等等")

    def test_event_serialization_is_deterministic_and_protocol_shaped(self):
        event = AssistantSpeechChunkEvent(
            event_id="evt-2",
            timestamp=124.0,
            source="tts-adapter",
            payload={"text": "你好"},
        )

        expected = {
            "event": "ASSISTANT_SPEECH_CHUNK",
            "event_id": "evt-2",
            "timestamp": 124.0,
            "source": "tts-adapter",
            "payload": {"text": "你好"},
        }
        self.assertEqual(event.to_dict(), expected)
        self.assertEqual(event.to_dict(), event.to_dict())

    def test_adapter_modules_export_abstract_async_contracts(self):
        adapters = (TurnAdapter, ASRAdapter, LLMAdapter, TTSAdapter, TranslationAdapter)
        for adapter in adapters:
            self.assertTrue(inspect.isabstract(adapter))

        self.assertTrue(inspect.iscoroutinefunction(TurnAdapter.push_audio))
        self.assertTrue(inspect.iscoroutinefunction(ASRAdapter.stream_audio))
        self.assertTrue(inspect.iscoroutinefunction(LLMAdapter.generate))
        self.assertTrue(inspect.iscoroutinefunction(LLMAdapter.stream_tokens))
        self.assertTrue(inspect.iscoroutinefunction(LLMAdapter.cancel))
        self.assertTrue(inspect.iscoroutinefunction(TTSAdapter.synthesize))
        self.assertTrue(inspect.iscoroutinefunction(TTSAdapter.stream_audio))
        self.assertTrue(inspect.iscoroutinefunction(TTSAdapter.interrupt))
        self.assertTrue(inspect.iscoroutinefunction(TranslationAdapter.translate_stream))

    def test_state_objects_can_be_constructed_with_shared_contract_fields(self):
        conversation = ConversationState(session_id="session-1")
        task = TaskState(task_type="counting", state={"current_number": 5})
        runtime = RuntimeState(
            active_workers=["worker-1"],
            model_status={"asr": "ready"},
            session_status="active",
        )

        self.assertEqual(asdict(conversation), {
            "session_id": "session-1",
            "current_mode": "LISTENING",
            "active_task": None,
            "current_response_id": None,
        })
        self.assertEqual(asdict(task), {
            "task_type": "counting",
            "state": {"current_number": 5},
        })
        self.assertEqual(asdict(runtime), {
            "active_workers": ["worker-1"],
            "model_status": {"asr": "ready"},
            "session_status": "active",
        })
