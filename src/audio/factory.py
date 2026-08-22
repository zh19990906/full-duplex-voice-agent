"""Configuration-driven creation of hardware-isolated audio adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.adapters.audio.microphone import MicrophoneAdapter
from src.adapters.audio.speaker import SpeakerAdapter
from src.adapters.audio.sounddevice import SoundDeviceInputBackend, SoundDeviceOutputBackend


def _scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.isdigit():
        return int(value)
    return value


def load_audio_config(path: str | Path = "configs/audio.yaml") -> dict[str, dict[str, Any]]:
    """Load the small, dependency-free audio YAML subset used by deployment."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    result: dict[str, dict[str, Any]] = {"input": {}, "output": {}}
    section: str | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            raise ValueError(f"invalid audio config at line {line_number}")
        key, value = (part.strip() for part in raw_line.strip().split(":", 1))
        if indentation == 0:
            if key != "audio" or value:
                raise ValueError(f"unsupported audio config entry at line {line_number}")
        elif indentation == 2 and key in result and not value:
            section = key
        elif indentation >= 4 and section is not None and value:
            result[section][key] = _scalar(value)
        else:
            raise ValueError(f"invalid audio config indentation at line {line_number}")
    return result


def _sections(config: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if config is None:
        return load_audio_config()
    if isinstance(config, (str, Path)):
        return load_audio_config(config)
    if "audio" in config and isinstance(config["audio"], Mapping):
        return config["audio"]
    return config


def _section(config: Mapping[str, Any] | str | Path | None, name: str) -> Mapping[str, Any]:
    values = _sections(config).get(name)
    if not isinstance(values, Mapping):
        raise ValueError(f"audio config must define {name} settings")
    return values


def _validate_provider(settings: Mapping[str, Any], kind: str) -> None:
    provider = settings.get("provider")
    if provider not in {"fake", "sounddevice", "pyaudio", "default"}:
        raise ValueError(f"unsupported audio {kind} provider: {provider}")


def create_audio_input(config: Mapping[str, Any] | str | Path | None = None, backend: Any | None = None) -> MicrophoneAdapter:
    settings = _section(config, "input")
    _validate_provider(settings, "input")
    if backend is None:
        if settings.get("provider") == "sounddevice":
            backend = SoundDeviceInputBackend(
                int(settings.get("sample_rate", 16000)),
                int(settings.get("channels", 1)),
                settings.get("device"),
            )
        else:
            raise RuntimeError("audio input backend must be injected for this provider")
    return MicrophoneAdapter(backend, int(settings.get("sample_rate", 16000)), int(settings.get("channels", 1)))


def create_audio_output(config: Mapping[str, Any] | str | Path | None = None, backend: Any | None = None) -> SpeakerAdapter:
    settings = _section(config, "output")
    _validate_provider(settings, "output")
    if backend is None:
        if settings.get("provider") == "sounddevice":
            backend = SoundDeviceOutputBackend(
                int(settings.get("sample_rate", 24000)),
                int(settings.get("channels", 1)),
                settings.get("device"),
            )
        else:
            raise RuntimeError("audio output backend must be injected for this provider")
    return SpeakerAdapter(backend)
