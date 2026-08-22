"""Provider-neutral model profile loading and local path resolution.

The project intentionally uses a small YAML subset here so deployment tooling
does not require PyYAML or any provider SDK.  The parser accepts the stable,
flat profile format in ``configs/models.yaml``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SUPPORTED_PROVIDERS = frozenset({
    "huggingface", "modelscope", "local", "whisper", "llama_cpp", "cosyvoice",
})
REQUIRED_FIELDS = frozenset({"provider", "model_id", "local_path"})


@dataclass(frozen=True)
class ModelProfile:
    """Configuration for one replaceable model profile."""

    name: str
    provider: str
    model_id: str
    local_path: str
    device: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Loaded deployment configuration."""

    model_root: str
    profiles: Mapping[str, ModelProfile]


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_model_config(path: str | Path) -> ModelConfig:
    """Load the supported model profile YAML subset using only the stdlib."""

    config_path = Path(path)
    lines = config_path.read_text(encoding="utf-8").splitlines()
    model_root = "./models"
    raw_profiles: dict[str, dict[str, str]] = {}
    current_name: str | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        if "\t" in raw_line[:indentation]:
            raise ValueError(f"tabs are not supported in {config_path}:{line_number}")
        if ":" not in raw_line:
            raise ValueError(f"invalid YAML entry at {config_path}:{line_number}")

        key, value = raw_line.strip().split(":", 1)
        key = key.strip()
        value = _parse_scalar(value)
        if indentation == 0:
            if key == "model_root":
                model_root = value
                current_name = None
            elif value == "":
                current_name = key
                raw_profiles[current_name] = {}
            else:
                raise ValueError(f"unsupported top-level value at {config_path}:{line_number}")
        elif current_name is not None and indentation >= 2:
            if not value:
                raise ValueError(f"empty profile field at {config_path}:{line_number}")
            raw_profiles[current_name][key] = value
        else:
            raise ValueError(f"invalid indentation at {config_path}:{line_number}")

    if Path(model_root).is_absolute():
        raise ValueError("model_root must be a relative path")

    profiles: dict[str, ModelProfile] = {}
    for name, values in raw_profiles.items():
        missing = REQUIRED_FIELDS - values.keys()
        if missing:
            raise ValueError(f"profile {name!r} is missing fields: {sorted(missing)}")
        provider = values["provider"].lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported provider {provider!r} for profile {name!r}")
        if Path(values["local_path"]).is_absolute():
            raise ValueError(f"local_path for {name!r} must be relative")
        profiles[name] = ModelProfile(
            name=name,
            provider=provider,
            model_id=values["model_id"],
            local_path=values["local_path"],
            device=values.get("device"),
        )
    return ModelConfig(model_root=model_root, profiles=profiles)


class ModelResolver:
    """Resolve configured profiles to local, provider-independent directories."""

    def __init__(
        self,
        config: ModelConfig | str | Path,
        model_home: str | Path | None = None,
    ) -> None:
        self.config = load_model_config(config) if isinstance(config, (str, Path)) else config
        configured_home = model_home or os.environ.get("MODEL_HOME")
        if configured_home:
            self.model_home = Path(configured_home).expanduser().resolve()
        else:
            config_root = Path(self.config.model_root)
            if config_root.is_absolute():
                raise ValueError("model_root must be relative")
            self.model_home = (Path.cwd() / config_root).resolve()

    def profile(self, name: str) -> ModelProfile:
        try:
            return self.config.profiles[name]
        except KeyError as exc:
            raise KeyError(f"unknown model profile: {name}") from exc

    def profiles(self, provider: str | None = None) -> Iterable[ModelProfile]:
        values = self.config.profiles.values()
        if provider is not None:
            provider = provider.lower()
            values = (profile for profile in values if profile.provider == provider)
        return values

    def _relative_path(self, profile: ModelProfile) -> Path:
        path = Path(profile.local_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"local_path for {profile.name!r} must stay relative")
        parts = path.parts
        if parts and parts[0] == "models":
            path = Path(*parts[1:])
        return path

    def resolve(self, name: str) -> Path:
        """Return the resolved path without creating it."""

        return (self.model_home / self._relative_path(self.profile(name))).resolve()

    def is_available(self, name: str) -> bool:
        return self.resolve(name).is_dir()

    def require_available(self, name: str) -> Path:
        path = self.resolve(name)
        if not path.is_dir():
            raise FileNotFoundError(
                f"model profile {name!r} is not available at {path}; prepare or copy the model first"
            )
        return path

    def prepare(self, name: str) -> Path:
        path = self.resolve(name)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Resolve configured model profile locations")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--provider")
    parser.add_argument("--model-home")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    resolver = ModelResolver(args.config, model_home=args.model_home)
    for profile in resolver.profiles(provider=args.provider):
        target = resolver.prepare(profile.name) if args.prepare else resolver.resolve(profile.name)
        print(f"{profile.name}: {target}")
        if profile.provider == "huggingface":
            print(f"  preview: hf download {profile.model_id} --local-dir {target}")
        elif profile.provider == "modelscope":
            print(f"  preview: modelscope download --model {profile.model_id} --local_dir {target}")
        else:
            print(f"  preview: copy local model files into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
