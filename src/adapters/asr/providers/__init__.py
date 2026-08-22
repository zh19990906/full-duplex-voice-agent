"""Optional ASR provider implementations behind the stable adapter boundary."""

from .whisper import WhisperASRProvider

__all__ = ["WhisperASRProvider"]
