"""Read-only health reporting for the production runtime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.model_runtime.lifecycle import ModelLifecycleManager, ModelState


class HealthChecker:
    """Inspect application and model state without changing either."""

    def __init__(
        self,
        application_state: Any,
        lifecycle_manager: ModelLifecycleManager,
        model_names: Iterable[str] = ("asr", "llm", "tts"),
    ) -> None:
        self.application_state = application_state
        self.lifecycle_manager = lifecycle_manager
        self.model_names = tuple(model_names)

    def health(self) -> dict[str, Any]:
        """Return a JSON-compatible health snapshot."""

        application_ready = self._application_ready()
        models: dict[str, str] = {}
        for name in self.model_names:
            try:
                models[name] = self.lifecycle_manager.status(name).value.lower()
            except KeyError:
                models[name] = "unknown"

        values = set(models.values())
        if not application_ready or "error" in values or "unknown" in values:
            overall = "unhealthy"
        elif all(value in {"ready", "active"} for value in values):
            overall = "healthy"
        else:
            overall = "degraded"
        return {
            "status": overall,
            "application": "ready" if application_ready else "stopped",
            "runtime": "available" if application_ready else "unavailable",
            "models": models,
        }

    def _application_ready(self) -> bool:
        value = self.application_state
        if callable(value):
            return bool(value())
        if isinstance(value, bool):
            return value
        return bool(getattr(value, "initialized", False))


__all__ = ["HealthChecker"]
