import unittest

from src.controller.states import ControllerState
from src.realtime.identifiers import GenerationClock, IdentifierAllocator
from src.realtime.session_state import (
    ConversationMode,
    FloorState,
    ResponseRecord,
    ResponseState,
    SessionState,
)


class RealtimeSessionStateTests(unittest.TestCase):
    def test_generation_clock_invalidates_old_epoch(self):
        """Catches a clock that does not advance or accepts stale work."""
        clock = GenerationClock()

        first = clock.current
        second = clock.advance()

        self.assertEqual(first, 0)
        self.assertEqual(second, 1)
        self.assertFalse(clock.is_current(first))
        self.assertTrue(clock.is_current(second))

    def test_identifier_allocator_produces_distinct_protocol_string_ids(self):
        """Catches ID reuse or identities incompatible with RealtimeEnvelope."""
        allocator = IdentifierAllocator()

        first_event = allocator.next_event_id()
        second_event = allocator.next_event_id()
        first_response = allocator.next_response_id()

        self.assertEqual(first_event, "evt-1")
        self.assertEqual(second_event, "evt-2")
        self.assertEqual(first_response, "response-1")
        self.assertIsInstance(first_event, str)
        self.assertIsInstance(first_response, str)

    def test_session_state_keeps_three_orthogonal_dimensions_and_act_metadata(self):
        """Catches conflating mode, floor, response, or adding assistant act as state."""
        state = SessionState(
            mode=ConversationMode.INTERPRETATION,
            floor=FloorState.OVERLAP,
            response=ResponseState.DUCKED,
            assistant_act="ASKING",
        )

        self.assertEqual(state.mode, ConversationMode.INTERPRETATION)
        self.assertEqual(state.floor, FloorState.OVERLAP)
        self.assertEqual(state.response, ResponseState.DUCKED)
        self.assertEqual(state.assistant_act, "ASKING")
        self.assertEqual(set(SessionState.__dataclass_fields__), {"mode", "floor", "response", "assistant_act"})

    def test_response_record_binds_response_to_its_generation_without_cursors(self):
        """Catches responses that cannot reject stale generation output."""
        record = ResponseRecord(response_id="response-4", generation_epoch=2)

        self.assertEqual(record.response_id, "response-4")
        self.assertEqual(record.generation_epoch, 2)
        self.assertEqual(record.state, ResponseState.GENERATING)
        self.assertEqual(set(ResponseRecord.__dataclass_fields__), {"response_id", "generation_epoch", "state"})

    def test_legacy_controller_state_remains_available(self):
        """Catches a compatibility break for existing controller imports."""
        self.assertEqual(ControllerState.SPEAKING.value, "SPEAKING")


if __name__ == "__main__":
    unittest.main()
