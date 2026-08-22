import unittest
from unittest.mock import patch

from src.adapters.turn.config import X2TurnConfig
from src.adapters.turn.x2_turn_adapter import (
    X2TurnDependencyError,
    X2TurnAdapter,
)
from src.core.events.events import (
    UserBackchannelEvent,
    UserInterruptEvent,
    UserSpeechPartialEvent,
    UserTurnEndEvent,
)


class FakeTurnBackend:
    def __init__(self, output):
        self.output = output
        self.received_audio = []

    def process_audio(self, audio_chunk):
        self.received_audio.append(audio_chunk)
        return self.output


class X2TurnAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_initializes_with_fake_backend_and_config(self):
        backend = FakeTurnBackend("backchannel")
        config = X2TurnConfig(
            model_path="models/turn",
            device="cpu",
            runtime_options={"threshold": 0.5},
        )
        adapter = X2TurnAdapter(config=config, backend=backend)

        self.assertIs(adapter.backend, backend)
        self.assertEqual(adapter.config, config)

    async def test_x2_turn_states_map_to_existing_events(self):
        expected = (
            ("backchannel", UserBackchannelEvent),
            ("interrupt", UserInterruptEvent),
            ("turn_end", UserTurnEndEvent),
            ("partial", UserSpeechPartialEvent),
        )

        for state, event_type in expected:
            adapter = X2TurnAdapter(backend=FakeTurnBackend(state))
            received = []
            adapter.event_handler = received.append

            await adapter.push_audio(b"audio")

            self.assertEqual(len(received), 1)
            self.assertIs(type(received[0]), event_type)

    async def test_push_audio_passes_bytes_to_backend_and_emits_payload(self):
        backend = FakeTurnBackend(
            {"state": "partial", "transcript": "上海", "confidence": 0.9}
        )
        received = []
        adapter = X2TurnAdapter(backend=backend, event_handler=received.append)

        await adapter.push_audio(b"chunk")

        self.assertEqual(backend.received_audio, [b"chunk"])
        self.assertEqual(received[0].payload, {"transcript": "上海", "confidence": 0.9})

    async def test_missing_dependency_has_clear_error(self):
        with patch(
            "src.adapters.turn.x2_turn_adapter.importlib.import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(X2TurnDependencyError, "x2_turn"):
                X2TurnAdapter()
