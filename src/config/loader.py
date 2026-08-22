"""Dependency-free configuration loading for environment profiles.

The repository configuration files intentionally use a small YAML subset:
nested mappings, scalar values, comments, and quoted strings.  Keeping the
parser here in the standard library avoids adding a runtime dependency merely
to select an environment profile.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or fails validation."""


def _strip_comment(value: str) -> str:
    """Remove a simple YAML comment while preserving quoted ``#`` values."""

    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
        elif character == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load the project-supported YAML mapping from ``path``."""

    config_path = Path(path)
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"unable to read configuration file {config_path}: {exc}") from exc

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(lines, start=1):
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]:
            raise ConfigError(f"tabs are not supported in {config_path}:{line_number}")
        content = _strip_comment(raw_line).strip()
        if not content:
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in content:
            raise ConfigError(f"invalid YAML entry at {config_path}:{line_number}")
        key, raw_value = (part.strip() for part in content.split(":", 1))
        if not key:
            raise ConfigError(f"empty YAML key at {config_path}:{line_number}")
        while stack and indentation <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"invalid indentation at {config_path}:{line_number}")
        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = _scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indentation, child))
    return root


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy where values in ``override`` take precedence."""

    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


class ConfigLoader:
    """Load and validate one named environment profile."""

    def __init__(self, config_dir: str | Path = "configs") -> None:
        self.config_dir = Path(config_dir)

    def load(self, profile: str = "dev") -> dict[str, Any]:
        """Load the base configuration and merge the named profile over it."""

        profile_path = self.config_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.is_file():
            raise ConfigError(f"unknown configuration profile: {profile}")
        merged = deep_merge(self._base_config(), load_yaml(profile_path))
        self._validate(merged, profile)
        return merged

    def load_from_env(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Load ``VOICE_AGENT_PROFILE`` or use the development profile."""

        environment = os.environ if environ is None else environ
        return self.load(environment.get("VOICE_AGENT_PROFILE", "dev"))

    def _base_config(self) -> dict[str, Any]:
        audio_document = load_yaml(self.config_dir / "audio.yaml")
        models_document = load_yaml(self.config_dir / "models.yaml")
        demo_document = load_yaml(self.config_dir / "demo.yaml")

        models = dict(models_document)
        base = {
            "audio": {},
            "models": {},
            "runtime": {
                "environment": "development",
                "scheduler": {"enabled": True},
                "workers": {"enabled": True},
            },
        }
        # The demo file is an older convenience config.  Load it first so the
        # detailed audio/model files remain authoritative for shared keys.
        base = deep_merge(base, demo_document)
        base = deep_merge(base, audio_document)
        base = deep_merge(base, {"models": models})
        return base

    @staticmethod
    def _validate(config: Mapping[str, Any], profile: str) -> None:
        for section in ("audio", "models", "runtime"):
            if not isinstance(config.get(section), Mapping):
                raise ConfigError(f"profile {profile!r} missing required section: {section}")

        models = config["models"]
        for model_name in ("asr", "llm", "tts"):
            model = models.get(model_name)
            if not isinstance(model, Mapping):
                raise ConfigError(f"profile {profile!r} missing required model section: models.{model_name}")
            provider = model.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                raise ConfigError(f"profile {profile!r} requires models.{model_name}.provider")


__all__ = ["ConfigError", "ConfigLoader", "deep_merge", "load_yaml"]
