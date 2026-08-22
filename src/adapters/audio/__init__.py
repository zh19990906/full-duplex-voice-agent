"""Replaceable hardware backends for the audio stream adapters."""

from .base import AudioInputBackend, AudioOutputBackend
from .microphone import MicrophoneAdapter
from .speaker import SpeakerAdapter
from .sounddevice import SoundDeviceInputBackend, SoundDeviceOutputBackend

__all__ = [
    "AudioInputBackend",
    "AudioOutputBackend",
    "MicrophoneAdapter",
    "SpeakerAdapter",
    "SoundDeviceInputBackend",
    "SoundDeviceOutputBackend",
]
