"""Side-effect-free controller action data objects."""

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Commands for external components to interpret and execute."""

    CONTINUE_GENERATION = "CONTINUE_GENERATION"
    STOP_RESPONSE = "STOP_RESPONSE"
    CANCEL_GENERATION = "CANCEL_GENERATION"
    PROCESS_USER_REQUEST = "PROCESS_USER_REQUEST"


@dataclass(frozen=True)
class ControllerAction:
    """An action description with no execution behavior."""

    action_type: ActionType
