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
        events = fusion.accept_turn(
            TurnCandidate("backchannel", 0.9, capture_timestamp=4.1, sequence=9, revision_id=10)
        )

        self.assertEqual(transcript_events[0].event, "USER_TRANSCRIPT_PARTIAL_CANDIDATE")
        self.assertEqual(events[0].event, "USER_BACKCHANNEL_CANDIDATE")
        self.assertEqual(events[0].timestamp, 4.1)
        self.assertEqual(events[0].payload["sequence"], 9)
        self.assertEqual(events[0].payload["revision_id"], 10)
        self.assertEqual(events[0].payload["transcript_evidence"], partial.to_dict())
        self.assertEqual(fusion.accept_turn(TurnCandidate("backchannel", 0.69)), ())

    def test_transcript_evidence_is_scoped_to_one_turn_across_reordered_signals(self):
        """Catches a new acoustic turn inheriting the preceding turn's ASR text."""
        fusion = SpeechEventFusion(turn_end_frames=2)
        first = TranscriptChunk("asr-1", "第一轮", 1.0, False, revision_id=1)

        # ASR may arrive before the first activity edge; opening activity must
        # preserve evidence already collected for this same turn.
        fusion.accept_transcript(first)
        fusion.accept_activity(AudioActivityCandidate(1, 1.1, True, 600.0))
        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.9)), ())
        ended = fusion.accept_turn(TurnCandidate("turn_end", 0.9))
        self.assertEqual(ended[0].payload["transcript_evidence"], first.to_dict())

        # A later acoustic-only candidate is not allowed to inherit first's text.
        after_close = fusion.accept_turn(TurnCandidate("backchannel", 0.9))
        self.assertIsNone(after_close[0].payload["transcript_evidence"])

        # Activity-first ordering opens a fresh turn, then its ASR is attached.
        fusion.accept_activity(AudioActivityCandidate(2, 2.0, False, 0.0))
        fusion.accept_activity(AudioActivityCandidate(3, 2.1, True, 650.0))
        second = TranscriptChunk("asr-2", "第二轮", 2.2, False, revision_id=2)
        fusion.accept_transcript(second)
        self.assertEqual(fusion.accept_turn(TurnCandidate("turn_end", 0.9)), ())
        next_ended = fusion.accept_turn(TurnCandidate("turn_end", 0.9))
        self.assertEqual(next_ended[0].payload["transcript_evidence"], second.to_dict())

    def test_backchannel_threshold_rejects_boolean_and_nonfinite_values(self):
        """Catches boolean or NaN thresholds silently changing backchannel policy."""
        for value in (True, False, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SpeechEventFusion(backchannel_confidence=value)

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
