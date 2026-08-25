"""Hysteretic fusion of acoustic, ASR, and X2 evidence into candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

from src.adapters.turn.x2_turn_streaming import TurnCandidate
from src.asr.stream import TranscriptChunk
from src.realtime.audio_ingress import AudioActivityCandidate
from src.realtime.session_state import SessionState


@dataclass(frozen=True)
class SpeechCandidateEvent:
    """Typed evidence event for the semantic policy layer, never an action."""

    event: str
    event_id: str
    timestamp: float
    source: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": dict(self.payload),
        }


class SpeechEventFusion:
    """Convert low-level observations to policy candidates with hysteresis."""

    def __init__(
        self,
        *,
        speaking_frames: int = 1,
        idle_frames: int = 1,
        turn_end_frames: int = 2,
        backchannel_confidence: float = 0.7,
        session_state: SessionState | None = None,
        assistant_context: Mapping[str, Any] | None = None,
    ) -> None:
        for name, value in {
            "speaking_frames": speaking_frames,
            "idle_frames": idle_frames,
            "turn_end_frames": turn_end_frames,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(backchannel_confidence, bool)
            or not isinstance(backchannel_confidence, (int, float))
            or not math.isfinite(float(backchannel_confidence))
            or not 0 <= backchannel_confidence <= 1
        ):
            raise ValueError("backchannel_confidence must be between 0 and 1")
        self.speaking_frames = speaking_frames
        self.idle_frames = idle_frames
        self.turn_end_frames = turn_end_frames
        self.backchannel_confidence = float(backchannel_confidence)
        self.session_state = session_state
        self.assistant_context = dict(assistant_context or {})
        self.latest_transcript: TranscriptChunk | None = None
        self._turn_open = False
        self._activity_active = False
        self._speech_latched = False
        self._turn_end_latched = False
        self._speaking_count = 0
        self._idle_count = 0
        self._turn_end_count = 0
        self._event_number = 0

    def accept_activity(self, candidate: AudioActivityCandidate) -> tuple[SpeechCandidateEvent, ...]:
        """Accept VAD-like energy evidence and publish only a rising edge."""
        if not isinstance(candidate, AudioActivityCandidate):
            raise TypeError("activity candidate must be AudioActivityCandidate")
        if not candidate.active:
            self._activity_active = False
            return ()
        if self._activity_active:
            return ()
        self._activity_active = True
        self._open_turn()
        self._speech_latched = True
        self._turn_end_latched = False
        self._turn_end_count = 0
        return (
            self._event(
                "USER_SPEECH_START_CANDIDATE",
                candidate.capture_timestamp,
                "audio_activity",
                {
                    "sequence": candidate.sequence,
                    "active": candidate.active,
                    "rms": candidate.rms,
                    "source_evidence": {
                        "sequence": candidate.sequence,
                        "capture_timestamp": candidate.capture_timestamp,
                        "active": candidate.active,
                        "rms": candidate.rms,
                    },
                },
            ),
        )

    def accept_transcript(self, chunk: TranscriptChunk) -> tuple[SpeechCandidateEvent, ...]:
        """Store ASR evidence and publish it without deciding user intent."""
        if not isinstance(chunk, TranscriptChunk):
            raise TypeError("transcript chunk must be TranscriptChunk")
        self._open_turn()
        self.latest_transcript = chunk
        name = "USER_TRANSCRIPT_FINAL_CANDIDATE" if chunk.is_final else "USER_TRANSCRIPT_PARTIAL_CANDIDATE"
        return (
            self._event(
                name,
                chunk.timestamp,
                "asr",
                {
                    "text": chunk.text,
                    "is_final": chunk.is_final,
                    "revision_id": chunk.revision_id,
                    "unstable_text": chunk.unstable_text,
                    "committed_text": chunk.committed_text,
                    "replaces_committed": chunk.replaces_committed,
                    "source_evidence": chunk.to_dict(),
                },
            ),
        )

    def accept_turn(self, candidate: TurnCandidate) -> tuple[SpeechCandidateEvent, ...]:
        """Apply acoustic hysteresis and return policy candidates only."""
        if not isinstance(candidate, TurnCandidate):
            raise TypeError("turn candidate must be TurnCandidate")
        label = candidate.label
        if label in {"speaking", "noidle"}:
            self._speaking_count += 1
            self._idle_count = 0
            self._turn_end_count = 0
            self._turn_end_latched = False
            if self._speaking_count >= self.speaking_frames and not self._speech_latched:
                self._open_turn()
                self._speech_latched = True
                return (self._turn_event("USER_SPEECH_START_CANDIDATE", candidate),)
            return ()
        if label == "idle":
            self._idle_count += 1
            self._speaking_count = 0
            self._turn_end_count = 0
            if self._idle_count >= self.idle_frames:
                self._speech_latched = False
            return ()
        if label == "turn_end":
            self._turn_end_count += 1
            self._speaking_count = 0
            self._idle_count = 0
            if self._turn_end_count >= self.turn_end_frames and not self._turn_end_latched:
                self._turn_end_latched = True
                event = self._turn_event("USER_TURN_END_CANDIDATE", candidate)
                self._close_turn()
                return (event,)
            return ()
        if label == "backchannel":
            self._speaking_count = 0
            self._idle_count = 0
            self._turn_end_count = 0
            if candidate.confidence >= self.backchannel_confidence:
                return (self._turn_event("USER_BACKCHANNEL_CANDIDATE", candidate),)
            return ()
        raise ValueError(f"unsupported fusion turn label: {label!r}")

    def reset_turn(self) -> None:
        """Discard current-turn transcript evidence without a semantic action."""
        self._close_turn()

    def _open_turn(self) -> None:
        if not self._turn_open:
            self._turn_open = True
            self.latest_transcript = None

    def _close_turn(self) -> None:
        self._turn_open = False
        self.latest_transcript = None
        self._speech_latched = False

    def _turn_event(self, name: str, candidate: TurnCandidate) -> SpeechCandidateEvent:
        return self._event(
            name,
            candidate.capture_timestamp if candidate.capture_timestamp is not None else time.time(),
            "x2_turn",
            {
                "label": candidate.label,
                "confidence": candidate.confidence,
                "sequence": candidate.sequence,
                "revision_id": candidate.revision_id,
                "inference_duration": candidate.inference_duration,
                "rtf": candidate.rtf,
                "metadata": dict(candidate.metadata),
                "source_evidence": candidate.to_dict(),
                "transcript_evidence": (
                    self.latest_transcript.to_dict() if self.latest_transcript is not None else None
                ),
            },
        )

    def _event(
        self, name: str, timestamp: float, source: str, payload: dict[str, Any]
    ) -> SpeechCandidateEvent:
        self._event_number += 1
        enriched = dict(payload)
        enriched["session"] = _session_snapshot(self.session_state)
        enriched["assistant_context"] = dict(self.assistant_context)
        return SpeechCandidateEvent(
            event=name,
            event_id=f"speech-candidate-{self._event_number}",
            timestamp=timestamp,
            source=source,
            payload=enriched,
        )


def _session_snapshot(state: SessionState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "mode": state.mode.value,
        "floor": state.floor.value,
        "response": state.response.value,
        "assistant_act": state.assistant_act,
    }
