import unittest

from src.adapters.turn.x2_turn_streaming import TurnCandidate
from src.asr.stream import TranscriptChunk
from src.core.events.events import (
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)
from src.realtime.audio_ingress import AudioActivityCandidate
from src.realtime.session_state import FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechEventFusion


class SpeechEventFusionTests(unittest.TestCase):
    def test_activity_emits_only_rising_speech_start_edges(self):
        """Catches one USER_SPEECH_START_CANDIDATE per active audio frame."""
        fusion = SpeechEventFusion()
        active = AudioActivityCandidate(1, 1.5, True, 700.0)
        inactive = AudioActivityCandidate(2, 1.52, False, 0.0)

        first = fusion.accept_activity(active)
        self.assertEqual([event.event for event in first], ["USER_SPEECH_START_CANDIDATE"])
        self.assertEqual(fusion.accept_activity(active), ())
        self.assertEqual(fusion.accept_activity(inactive), ())
        self.assertEqual(
            [event.event for event in fusion.accept_activity(active)],
            ["USER_SPEECH_START_CANDIDATE"],
        )

    def test_turn_hysteresis_resets_on_contrary_labels_and_latches_until_new_speech(self):
        """Catches a single noisy turn-end frame completing or repeatedly ending a turn."""
        fusion = SpeechEventFusion(speaking_frames=2, idle_frames=2, turn_end_frames=2)

        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.8)), ())
        self.assertEqual(fusion.accept_turn(TurnCandidate("speaking", 0.8)), ())
        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.8)), ())
        ended = fusion.accept_turn(TurnCandidate("turn_end", 0.9))
        self.assertEqual([event.event for event in ended], ["USER_TURN_END_CANDIDATE"])
        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.9)), ())

        self.assertEqual(fusion.accept_turn(TurnCandidate("speaking", 0.9)), ())
        speaking = fusion.accept_turn(TurnCandidate("speaking", 0.9))
        self.assertEqual([event.event for event in speaking], ["USER_SPEECH_START_CANDIDATE"])
        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.9)), ())
        self.assertEqual(
            [event.event for event in fusion.accept_turn(TurnCandidate("turn_end", 0.9))],
            ["USER_TURN_END_CANDIDATE"],
        )

    def test_backchannel_is_immediate_candidate_with_revision_aware_transcript_evidence(self):
        """Catches deterministic attachment of affirmations to final semantic actions."""
        fusion = SpeechEventFusion(backchannel_confidence=0.7)
        partial = TranscriptChunk(
            "asr-4",
            "对",
            4.0,
            False,
            revision_id=4,
            unstable_text="的",
        )
        transcript_events = fusion.accept_transcript(partial)
        events = fusion.accept_turn(TurnCandidate("backchannel", 0.9, capture_timestamp=4.1))

        self.assertEqual(transcript_events[0].event, "USER_TRANSCRIPT_PARTIAL_CANDIDATE")
        self.assertEqual(events[0].event, "USER_BACKCHANNEL_CANDIDATE")
        self.assertEqual(events[0].payload["transcript_evidence"], partial.to_dict())
        self.assertEqual(fusion.accept_turn(TurnCandidate("backchannel", 0.69)), ())

    def test_final_revision_and_context_remain_candidates_without_mutating_session_state(self):
        """Catches fusion changing policy state or emitting legacy semantic User* events."""
        state = SessionState(floor=FloorState.ASSISTANT, response=ResponseState.PLAYING)
        before = SessionState(state.mode, state.floor, state.response, state.assistant_act)
        fusion = SpeechEventFusion(session_state=state, assistant_context={"asking": True})
        final = TranscriptChunk(
            "asr-5",
            "北京",
            5.0,
            True,
            revision_id=5,
            committed_text="北京",
            replaces_committed=True,
        )

        events = fusion.accept_transcript(final)

        self.assertEqual(events[0].event, "USER_TRANSCRIPT_FINAL_CANDIDATE")
        self.assertEqual(events[0].payload["replaces_committed"], True)
        self.assertEqual(events[0].payload["session"], {
            "mode": "CHAT",
            "floor": "ASSISTANT",
            "response": "PLAYING",
            "assistant_act": None,
        })
        self.assertEqual(state, before)
        self.assertNotIsInstance(events[0], (UserBackchannelEvent, UserInterruptEvent, UserTurnEndEvent))


if __name__ == "__main__":
    unittest.main()
