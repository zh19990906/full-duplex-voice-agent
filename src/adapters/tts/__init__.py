"""Abstract streaming TTS adapter interfaces."""

from .base import BaseTTSAdapter
from .backend import StreamingTTSAdapter, StreamingTTSBackend

__all__ = ["BaseTTSAdapter", "StreamingTTSAdapter", "StreamingTTSBackend"]
