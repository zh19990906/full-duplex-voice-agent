"""Version 1 realtime event and browser PCM frame contracts."""

from dataclasses import dataclass
import struct
from typing import Any


PROTOCOL_VERSION = 1
PCM16_SAMPLE_RATE = 16000
PCM16_MONO_CHANNELS = 1
PCM16_FRAME_SAMPLES = 320
PCM16_FRAME_BYTES = PCM16_FRAME_SAMPLES * 2

# Network byte order: protocol version, sequence, capture timestamp,
# sample rate, and channel count.
_AUDIO_FRAME_HEADER = struct.Struct("!BIdIB")


@dataclass(frozen=True)
class RealtimeEnvelope:
    """Typed, versioned metadata carried by every V1 realtime event."""

    event: str
    event_id: str
    session_id: str
    sequence: int
    capture_timestamp: float
    server_timestamp: float
    response_id: str | None
    generation_epoch: int
    payload: dict[str, Any]
    segment_id: str | None = None
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable V1 transport representation."""
        return {
            "protocol_version": self.protocol_version,
            "event": self.event,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "capture_timestamp": self.capture_timestamp,
            "server_timestamp": self.server_timestamp,
            "response_id": self.response_id,
            "generation_epoch": self.generation_epoch,
            "segment_id": self.segment_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AudioFrameHeader:
    """Metadata prepended to each browser PCM16 audio frame."""

    sequence: int
    capture_timestamp: float
    sample_rate: int = PCM16_SAMPLE_RATE
    channels: int = PCM16_MONO_CHANNELS


def encode_audio_frame(header: AudioFrameHeader, pcm: bytes) -> bytes:
    """Encode a validated PCM16 frame using the V1 binary layout."""
    _validate_audio_contract(header, pcm)
    return _AUDIO_FRAME_HEADER.pack(
        PROTOCOL_VERSION,
        header.sequence,
        header.capture_timestamp,
        header.sample_rate,
        header.channels,
    ) + pcm


def decode_audio_frame(frame: bytes) -> tuple[AudioFrameHeader, bytes]:
    """Decode and validate a V1 browser PCM16 frame."""
    if len(frame) < _AUDIO_FRAME_HEADER.size:
        raise ValueError("truncated audio frame header")

    protocol_version, sequence, capture_timestamp, sample_rate, channels = (
        _AUDIO_FRAME_HEADER.unpack_from(frame)
    )
    if protocol_version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {protocol_version}")

    pcm = frame[_AUDIO_FRAME_HEADER.size :]
    header = AudioFrameHeader(sequence, capture_timestamp, sample_rate, channels)
    _validate_audio_contract(header, pcm)
    return header, pcm


def _validate_audio_contract(header: AudioFrameHeader, pcm: bytes) -> None:
    if header.sample_rate != PCM16_SAMPLE_RATE:
        raise ValueError(f"unsupported sample rate: {header.sample_rate}")
    if header.channels != PCM16_MONO_CHANNELS:
        raise ValueError(f"unsupported channel count: {header.channels}")
    if len(pcm) % 2:
        raise ValueError("PCM16 payload must contain whole samples")
    if len(pcm) != PCM16_FRAME_BYTES:
        raise ValueError(f"PCM16 payload must be exactly {PCM16_FRAME_BYTES} bytes")
