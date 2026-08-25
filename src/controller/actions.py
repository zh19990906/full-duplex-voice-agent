"""Side-effect-free controller action data objects."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class ActionType(str, Enum):
    """Commands for external components to interpret and execute."""

    CONTINUE_GENERATION = "CONTINUE_GENERATION"
    STOP_RESPONSE = "STOP_RESPONSE"
    CANCEL_GENERATION = "CANCEL_GENERATION"
    PROCESS_USER_REQUEST = "PROCESS_USER_REQUEST"
    DUCK_RESPONSE = "DUCK_RESPONSE"
    RESTORE_RESPONSE = "RESTORE_RESPONSE"
    PAUSE_RESPONSE = "PAUSE_RESPONSE"
    RESUME_RESPONSE = "RESUME_RESPONSE"
    REVISE_RESPONSE = "REVISE_RESPONSE"
    SWITCH_MODE = "SWITCH_MODE"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"


@dataclass(frozen=True)
class ControllerAction:
    """An action description with no execution behavior."""

    action_type: ActionType
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, ActionType):
            raise TypeError("action_type must be ActionType")
        if self.payload is not None:
            if not isinstance(self.payload, Mapping):
                raise TypeError("payload must be a mapping or None")
            object.__setattr__(self, "payload", _freeze_payload(self.payload))


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_payload(item) for item in value)
    return value
