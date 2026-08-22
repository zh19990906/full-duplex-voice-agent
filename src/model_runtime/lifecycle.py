"""Provider-neutral model resource lifecycle management."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ModelState(str, Enum):
    REGISTERED = "REGISTERED"
    LOADING = "LOADING"
    READY = "READY"
    WARMING = "WARMING"
    ACTIVE = "ACTIVE"
    UNLOADING = "UNLOADING"
    UNLOADED = "UNLOADED"
    ERROR = "ERROR"


@dataclass
class ModelRecord:
    name: str
    provider: str
    model_path: str
    resource: Any | None = None
    auto_load: bool = False
    state: ModelState = ModelState.REGISTERED
    error: str | None = None

    @property
    def status(self) -> ModelState:
        """Compatibility alias for callers that use status terminology."""

        return self.state


_TRANSITIONS = {
    ModelState.REGISTERED: {ModelState.LOADING},
    ModelState.LOADING: {ModelState.READY, ModelState.ERROR},
    ModelState.READY: {ModelState.WARMING, ModelState.UNLOADING},
    ModelState.WARMING: {ModelState.ACTIVE, ModelState.ERROR},
    ModelState.ACTIVE: {ModelState.UNLOADING},
    ModelState.UNLOADING: {ModelState.UNLOADED, ModelState.ERROR},
    ModelState.UNLOADED: {ModelState.LOADING},
    ModelState.ERROR: {ModelState.LOADING},
}


class ModelLifecycleManager:
    """Own model resource state without knowing how inference is performed."""

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}

    def register(
        self,
        name: str,
        provider: str | Any,
        model_path: str | Path,
        resource: Any | None = None,
        auto_load: bool = False,
    ) -> ModelRecord:
        if name in self._models:
            raise ValueError(f"model {name!r} is already registered")
        path = Path(model_path)
        if path.is_absolute():
            raise ValueError("model_path must be relative")
        provider_name = provider if isinstance(provider, str) else type(provider).__name__
        record = ModelRecord(name, provider_name, str(path), resource, auto_load)
        self._models[name] = record
        return record

    async def load(self, name: str) -> ModelRecord:
        record = self._record(name)
        self._transition(record, ModelState.LOADING)
        try:
            result = await self._invoke(record.resource, "load")
            if result is not None:
                record.resource = result
            self._transition(record, ModelState.READY)
        except Exception as exc:
            record.error = str(exc)
            record.state = ModelState.ERROR
            raise
        return record

    async def warmup(self, name: str) -> ModelRecord:
        record = self._record(name)
        self._transition(record, ModelState.WARMING)
        try:
            await self._invoke(record.resource, "warmup")
            self._transition(record, ModelState.ACTIVE)
        except Exception as exc:
            record.error = str(exc)
            record.state = ModelState.ERROR
            raise
        return record

    async def unload(self, name: str) -> ModelRecord:
        record = self._record(name)
        self._transition(record, ModelState.UNLOADING)
        try:
            await self._invoke(record.resource, "unload")
            self._transition(record, ModelState.UNLOADED)
        except Exception as exc:
            record.error = str(exc)
            record.state = ModelState.ERROR
            raise
        return record

    def status(self, name: str) -> ModelState:
        return self._record(name).state

    def record(self, name: str) -> ModelRecord:
        return self._record(name)

    def _record(self, name: str) -> ModelRecord:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"unknown model: {name}") from exc

    @staticmethod
    def _transition(record: ModelRecord, target: ModelState) -> None:
        if target not in _TRANSITIONS[record.state]:
            raise RuntimeError(
                f"invalid model state transition for {record.name!r}: "
                f"{record.state.value} -> {target.value}"
            )
        record.state = target
        if target is not ModelState.ERROR:
            record.error = None

    @staticmethod
    async def _invoke(resource: Any | None, method_name: str) -> Any | None:
        method = getattr(resource, method_name, None)
        if not callable(method):
            return None
        result = method()
        if inspect.isawaitable(result):
            return await result
        return result
