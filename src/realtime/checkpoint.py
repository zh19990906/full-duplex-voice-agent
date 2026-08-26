"""Pause/resume checkpoints for realtime assistant responses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResumePlan:
    """Instructions for replaying a paused response from a stable segment start."""

    response_id: str
    generation_epoch: int
    segment_id: int
    sample_offset: int
    audio: bytes
    text: str = ""
    playback_attempt_id: int | None = None


@dataclass
class _SegmentCheckpoint:
    segment_id: int
    text: str = ""
    audio: bytes = b""
    played_offset: int = 0

    @property
    def sample_count(self) -> int:
        return len(self.audio) // 2

    @property
    def sample_count_unknown(self) -> bool:
        return bool(self.audio) and len(self.audio) % 2 == 1


@dataclass
class ResponseCheckpoint:
    """Mutable in-memory state for one generated assistant response."""

    response_id: str
    generation_epoch: int
    generated_cursor: int = 0
    committed_cursor: int = 0
    synthesized_cursor: int = 0
    played_cursor: int = 0
    paused: bool = False
    archived: bool = False
    resumable: bool = True
    generated_text: str = ""
    segments: dict[int, _SegmentCheckpoint] = field(default_factory=dict)


class ResponseCheckpointStore:
    """Track replayable response state with monotonic text/audio cursors."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, ResponseCheckpoint] = {}

    def activate(self, response_id: str, *, generation_epoch: int) -> ResponseCheckpoint:
        checkpoint = ResponseCheckpoint(response_id=response_id, generation_epoch=generation_epoch)
        self._checkpoints[response_id] = checkpoint
        return checkpoint

    def get(self, response_id: str) -> ResponseCheckpoint | None:
        return self._checkpoints.get(response_id)

    def record_generated_text(self, response_id: str, text: str) -> None:
        checkpoint = self._require(response_id)
        checkpoint.generated_text += text
        checkpoint.generated_cursor += len(text)

    def record_segment(
        self,
        response_id: str,
        segment_id: int,
        text: str,
        *,
        audio: bytes = b"",
    ) -> None:
        checkpoint = self._require(response_id)
        segment = checkpoint.segments.get(segment_id)
        if segment is None:
            segment = _SegmentCheckpoint(segment_id=segment_id)
            checkpoint.segments[segment_id] = segment
        if text:
            segment.text = text
        if audio:
            segment.audio += audio
            checkpoint.synthesized_cursor += len(audio) // 2
        checkpoint.committed_cursor = max(
            checkpoint.committed_cursor,
            sum(len(item.text) for item in checkpoint.segments.values()),
        )

    def ack(self, response_id: str, *, segment_id: int, sample_offset: int) -> None:
        checkpoint = self._require(response_id)
        segment = checkpoint.segments.get(segment_id)
        if segment is None:
            return
        segment.played_offset = max(segment.played_offset, sample_offset)
        checkpoint.played_cursor = max(
            checkpoint.played_cursor,
            sum(min(item.played_offset, item.sample_count) for item in checkpoint.segments.values()),
        )

    def pause(self, response_id: str) -> ResponseCheckpoint:
        checkpoint = self._require(response_id)
        checkpoint.paused = True
        return checkpoint

    def resume(self, response_id: str) -> ResumePlan | None:
        checkpoint = self._require(response_id)
        if checkpoint.archived or not checkpoint.resumable:
            return None
        for segment_id in sorted(checkpoint.segments):
            segment = checkpoint.segments[segment_id]
            if segment.sample_count == 0 and segment.text:
                return ResumePlan(
                    response_id=response_id,
                    generation_epoch=checkpoint.generation_epoch,
                    segment_id=segment.segment_id,
                    sample_offset=0,
                    audio=segment.audio,
                    text=segment.text,
                )
            if segment.played_offset > 0 and segment.sample_count_unknown:
                return ResumePlan(
                    response_id=response_id,
                    generation_epoch=checkpoint.generation_epoch,
                    segment_id=segment.segment_id,
                    sample_offset=0,
                    audio=segment.audio,
                    text=segment.text,
                )
            if segment.played_offset < segment.sample_count:
                return ResumePlan(
                    response_id=response_id,
                    generation_epoch=checkpoint.generation_epoch,
                    segment_id=segment.segment_id,
                    sample_offset=0,
                    audio=segment.audio,
                    text=segment.text,
                )
        return None

    def discard_unplayed(self, response_id: str) -> ResponseCheckpoint:
        checkpoint = self._require(response_id)
        checkpoint.paused = False
        checkpoint.resumable = False
        return checkpoint

    def archive(self, response_id: str) -> ResponseCheckpoint:
        checkpoint = self.discard_unplayed(response_id)
        checkpoint.archived = True
        return checkpoint

    def _require(self, response_id: str) -> ResponseCheckpoint:
        checkpoint = self.get(response_id)
        if checkpoint is None:
            raise KeyError(f"unknown response checkpoint: {response_id}")
        return checkpoint


__all__ = ["ResponseCheckpoint", "ResponseCheckpointStore", "ResumePlan"]
