"""Translate optional X2-Turn output into project events."""

import importlib
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
    UserSpeechPartialEvent,
    UserTurnEndEvent,
)
from src.core.interfaces.turn import TurnAdapter

from .config import X2TurnConfig
from .x2_turn_streaming import TurnCandidate, normalize_turn_candidates


EventHandler = Callable[[BaseEvent], Awaitable[None] | None]


class X2TurnDependencyError(ImportError):
    """Raised when the optional X2-Turn backend cannot be loaded."""


class X2TurnAdapter(TurnAdapter):
    """Keep X2-Turn internals private and emit stable project events."""

    _STATE_EVENTS = {
        "backchannel": UserBackchannelEvent,
        "interrupt": UserInterruptEvent,
        "turn_end": UserTurnEndEvent,
        "partial": UserSpeechPartialEvent,
        "partial_speech": UserSpeechPartialEvent,
        "speech_partial": UserSpeechPartialEvent,
    }

    def __init__(
        self,
        config: X2TurnConfig | None = None,
        backend: Any | None = None,
        event_handler: EventHandler | None = None,
    ) -> None:
        self.config = config or X2TurnConfig()
        self.backend = backend if backend is not None else self._load_backend()
        self.event_handler = event_handler
        self.emitted_events: list[BaseEvent] = []
        self._event_number = 0

    async def push_audio(self, audio_chunk: bytes) -> None:
        """Pass audio to X2-Turn and emit translated events."""
        result = self._call_backend(audio_chunk)
        if inspect.isawaitable(result):
            result = await result

        outputs = result if isinstance(result, (list, tuple)) else [result]
        for output in outputs:
            if output is None:
                continue
            event = self.map_output(output)
            self.emitted_events.append(event)
            if self.event_handler is not None:
                handled = self.event_handler(event)
                if inspect.isawaitable(handled):
                    await handled

    def map_output(self, output: Any) -> BaseEvent:
        """Map one backend result without exposing its original structure."""
        state, payload, timestamp = self._normalize_output(output)
        event_type = self._STATE_EVENTS.get(state)
        if event_type is None:
            raise ValueError(f"Unsupported X2-Turn state: {state!r}")
        self._event_number += 1
        return event_type(
            event_id=f"x2-turn-{self._event_number}",
            timestamp=timestamp,
            source="x2-turn",
            payload=payload,
        )

    def map_candidate(self, output: Any) -> TurnCandidate:
        """Map exactly one candidate; use :meth:`map_candidates` for many."""
        candidates = self.map_candidates(output)
        if len(candidates) != 1:
            raise ValueError(
                "X2-Turn output normalized to zero or multiple candidates; "
                "use map_candidates()"
            )
        return candidates[0]

    def map_candidates(self, output: Any) -> tuple[TurnCandidate, ...]:
        """Map official X2 wrapper evidence without emitting a legacy User* event."""
        return normalize_turn_candidates(output)

    def _call_backend(self, audio_chunk: bytes) -> Any:
        for method_name in ("process_audio", "push_audio", "infer"):
            method = getattr(self.backend, method_name, None)
            if method is not None:
                return method(audio_chunk)
        raise RuntimeError(
            "X2-Turn backend must provide process_audio(), push_audio(), or infer()."
        )

    @classmethod
    def _normalize_output(cls, output: Any) -> tuple[str, dict[str, Any], float]:
        if isinstance(output, str):
            state = output
            values: dict[str, Any] = {}
        elif isinstance(output, Mapping):
            values = dict(output)
            state = values.pop("state", values.pop("event", values.pop("type", None)))
        else:
            state = getattr(output, "state", None)
            values = {
                key: getattr(output, key)
                for key in ("transcript", "text", "confidence", "timestamp")
                if hasattr(output, key)
            }
        if not isinstance(state, str):
            raise ValueError("X2-Turn output must include a string state.")

        timestamp = values.pop("timestamp", None)
        if not isinstance(timestamp, (int, float)):
            timestamp = time.time()
        payload = {
            key: values[key]
            for key in ("transcript", "text", "confidence")
            if key in values
        }
        return state.strip().lower().replace("-", "_").replace(" ", "_"), payload, float(timestamp)

    def _load_backend(self) -> Any:
        try:
            module = importlib.import_module("x2_turn")
        except ImportError as exc:
            raise X2TurnDependencyError(
                "X2-Turn dependency 'x2_turn' is unavailable. Install it or "
                "inject a backend explicitly for tests."
            ) from exc

        for factory_name in ("create", "create_adapter", "X2Turn", "TurnDetector"):
            factory = getattr(module, factory_name, None)
            if callable(factory):
                return factory(**self.config.backend_options())
        raise X2TurnDependencyError(
            "X2-Turn dependency is installed but exposes no supported backend "
            "factory (create, create_adapter, X2Turn, or TurnDetector)."
        )
