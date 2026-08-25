"""Versioned contracts for the realtime runtime."""

from .protocol import (
    AudioFrameHeader,
    RealtimeEnvelope,
    decode_audio_frame,
    encode_audio_frame,
)
from .cancellation import ActiveTaskSlot, CancellationToken
from .identifiers import GenerationClock, IdentifierAllocator
from .response_pipeline import RealtimeResponsePipeline, ResponsePipelineResult
from .session_state import (
    ConversationMode,
    FloorState,
    ResponseRecord,
    ResponseState,
    SessionState,
)
from .text_segmenter import LanguageAwareTextSegmenter, TextSegment

__all__ = [
    "AudioFrameHeader",
    "ActiveTaskSlot",
    "CancellationToken",
    "ConversationMode",
    "FloorState",
    "GenerationClock",
    "IdentifierAllocator",
    "LanguageAwareTextSegmenter",
    "RealtimeResponsePipeline",
    "RealtimeEnvelope",
    "ResponseRecord",
    "ResponsePipelineResult",
    "ResponseState",
    "SessionState",
    "TextSegment",
    "decode_audio_frame",
    "encode_audio_frame",
]
