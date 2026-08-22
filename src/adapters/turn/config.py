"""Configuration for the optional X2-Turn backend."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class X2TurnConfig:
    """Backend settings without machine-specific defaults."""

    model_path: str | None = None
    device: str | None = None
    runtime_options: dict[str, Any] = field(default_factory=dict)

    def backend_options(self) -> dict[str, Any]:
        """Return constructor options for the isolated backend loader."""
        options = dict(self.runtime_options)
        if self.model_path is not None:
            options["model_path"] = self.model_path
        if self.device is not None:
            options["device"] = self.device
        return options
