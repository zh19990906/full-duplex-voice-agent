"""Optional TTS provider implementations behind the stable adapter boundary."""

from .cosyvoice import CosyVoiceTTSProvider
from .cosyvoice_worker import CosyVoiceWorkerClient

__all__ = ["CosyVoiceTTSProvider", "CosyVoiceWorkerClient"]
