"""Pure event-to-state/action conversation controller."""

from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)

from .actions import ActionType, ControllerAction
from .states import ControllerState


class ConversationController:
    """Apply deterministic state transitions without executing actions."""

    def __init__(self, state: ControllerState = ControllerState.IDLE) -> None:
        self.state = state

    @property
    def current_state(self) -> ControllerState:
        """Return the current state using an explicit read-only alias."""
        return self.state

    def handle_event(self, event: BaseEvent) -> tuple[ControllerAction, ...]:
        """Transform one event into state changes and action descriptions."""
        if isinstance(event, UserBackchannelEvent):
            if self.state is ControllerState.SPEAKING:
                return (ControllerAction(ActionType.CONTINUE_GENERATION),)
            return ()

        if isinstance(event, UserInterruptEvent):
            self.state = ControllerState.INTERRUPTED
            return (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
            )

        if isinstance(event, UserTurnEndEvent):
            self.state = ControllerState.THINKING
            return (ControllerAction(ActionType.PROCESS_USER_REQUEST),)

        return ()
