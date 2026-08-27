"""VRAM budget accounting for production model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable


class ModelMemoryBudgetError(RuntimeError):
    """Raised when a model would exceed the configured safe VRAM budget."""


@dataclass
class _ModelReservation:
    requested_bytes: int
    overhead_bytes: dict[str, int] = field(default_factory=dict)
    before_bytes: int | None = None
    after_bytes: int | None = None
    used_bytes: int | None = None

    @property
    def planned_bytes(self) -> int:
        return self.requested_bytes + sum(self.overhead_bytes.values())


class ModelManager:
    """Track configured VRAM headroom and per-model load diagnostics."""

    def __init__(
        self,
        *,
        total_vram_bytes: int,
        reserve_bytes: int = 0,
        measure_allocated_bytes: Callable[[], int] | None = None,
    ) -> None:
        for name, value in {
            "total_vram_bytes": total_vram_bytes,
            "reserve_bytes": reserve_bytes,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if reserve_bytes > total_vram_bytes:
            raise ValueError("reserve_bytes must not exceed total_vram_bytes")
        self.total_vram_bytes = total_vram_bytes
        self.reserve_bytes = reserve_bytes
        self.measure_allocated_bytes = measure_allocated_bytes
        self._loaded_bytes = 0
        self._models: dict[str, _ModelReservation] = {}
        self._lock = threading.RLock()

    def reserve(
        self,
        name: str,
        *,
        requested_bytes: int,
        overhead_bytes: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if type(requested_bytes) is not int or requested_bytes < 0:
            raise ValueError("requested_bytes must be a non-negative integer")
        normalized_overhead = {
            key: int(value)
            for key, value in dict(overhead_bytes or {}).items()
        }
        if any(value < 0 for value in normalized_overhead.values()):
            raise ValueError("overhead_bytes values must be non-negative")
        with self._lock:
            if name in self._models:
                raise ValueError(f"model already reserved: {name}")
            reservation = _ModelReservation(
                requested_bytes=requested_bytes,
                overhead_bytes=normalized_overhead,
                before_bytes=self._measure(),
            )
            available = (
                self.total_vram_bytes
                - self.reserve_bytes
                - self._loaded_bytes
                - self._reserved_bytes()
            )
            if reservation.planned_bytes > available:
                raise ModelMemoryBudgetError(
                    f"{name} exceeds safe VRAM budget: requested={reservation.planned_bytes} "
                    f"available={available}"
                )
            self._models[name] = reservation
            return self._diagnostic_record(name, reservation)

    def record_loaded(
        self, name: str, *, used_bytes: int | None = None
    ) -> dict[str, Any]:
        if used_bytes is not None and (type(used_bytes) is not int or used_bytes < 0):
            raise ValueError("used_bytes must be a non-negative integer")
        with self._lock:
            reservation = self._models.get(name)
            if reservation is None:
                if used_bytes is None:
                    raise ValueError("used_bytes is required without a prior reservation")
                reservation = _ModelReservation(requested_bytes=used_bytes)
                self._models[name] = reservation
            after_bytes = self._measure()
            measured_bytes = None
            if after_bytes is not None and reservation.before_bytes is not None:
                measured_bytes = max(0, after_bytes - reservation.before_bytes)
            actual_bytes = measured_bytes if measured_bytes is not None else used_bytes
            if actual_bytes is None:
                raise ValueError("used_bytes is required when CUDA measurement is unavailable")
            previous_used = reservation.used_bytes or 0
            other_pending = sum(
                item.planned_bytes
                for model_name, item in self._models.items()
                if model_name != name and item.used_bytes is None
            )
            projected = self._loaded_bytes - previous_used + actual_bytes + other_pending
            safe_capacity = self.total_vram_bytes - self.reserve_bytes
            reservation.after_bytes = after_bytes
            if projected > safe_capacity:
                raise ModelMemoryBudgetError(
                    f"{name} exceeds safe VRAM budget after load: measured={actual_bytes} "
                    f"projected={projected} available={safe_capacity}"
                )
            self._loaded_bytes += actual_bytes - previous_used
            reservation.used_bytes = actual_bytes
            return self._diagnostic_record(name, reservation)

    def release(self, name: str) -> bool:
        """Release one pending or loaded reservation exactly once."""
        with self._lock:
            reservation = self._models.pop(name, None)
            if reservation is None:
                return False
            if reservation.used_bytes is not None:
                self._loaded_bytes = max(0, self._loaded_bytes - reservation.used_bytes)
            return True

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_vram_bytes": self.total_vram_bytes,
                "reserve_bytes": self.reserve_bytes,
                "reserved_bytes": self._reserved_bytes(),
                "loaded_bytes": self._loaded_bytes,
                "models": {
                    name: self._diagnostic_record(name, reservation)
                    for name, reservation in self._models.items()
                },
            }

    def _reserved_bytes(self) -> int:
        return sum(
            reservation.planned_bytes
            for reservation in self._models.values()
            if reservation.used_bytes is None
        )

    def _measure(self) -> int | None:
        if self.measure_allocated_bytes is None:
            return None
        value = self.measure_allocated_bytes()
        if type(value) is not int or value < 0:
            raise ValueError("measure_allocated_bytes must return a non-negative integer")
        return value

    @staticmethod
    def _diagnostic_record(name: str, reservation: _ModelReservation) -> dict[str, Any]:
        return {
            "name": name,
            "requested_bytes": reservation.requested_bytes,
            "overhead_bytes": dict(reservation.overhead_bytes),
            "planned_bytes": reservation.planned_bytes,
            "used_bytes": reservation.used_bytes,
            "before_bytes": reservation.before_bytes,
            "after_bytes": reservation.after_bytes,
        }
