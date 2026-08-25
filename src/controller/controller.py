"""Pure event-to-state/action conversation controller."""

from src.core.events.events import (
    BaseEvent,
    UserBackchannelEvent,
    UserInterruptEvent,
    UserTurnEndEvent,
)

from .actions import ActionType, ControllerAction
from .states import ControllerState
from src.realtime.policy import PolicyAction, PolicyDecision
from src.realtime.session_state import ConversationMode, FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent


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

    def handle_candidate(
        self,
        state: SessionState,
        candidate: SpeechCandidateEvent,
    ) -> tuple[ControllerAction, ...]:
        """Apply reversible low-latency behavior before semantic confirmation."""
        if not isinstance(state, SessionState):
            raise TypeError("state must be SessionState")
        if not isinstance(candidate, SpeechCandidateEvent):
            raise TypeError("candidate must be SpeechCandidateEvent")
        if state.response not in {ResponseState.PLAYING, ResponseState.DUCKED}:
            return ()
        if candidate.event in {
            "USER_BACKCHANNEL_CANDIDATE",
            "USER_SPEECH_START_CANDIDATE",
        } and state.response is ResponseState.PLAYING:
            state.floor = FloorState.OVERLAP
            state.response = ResponseState.DUCKED
            return (ControllerAction(ActionType.DUCK_RESPONSE),)
        if candidate.event == "USER_TURN_END_CANDIDATE":
            state.floor = FloorState.OVERLAP
            state.response = ResponseState.PAUSED
            return (ControllerAction(ActionType.PAUSE_RESPONSE),)
        return ()

    def apply_policy(
        self,
        state: SessionState,
        policy: PolicyDecision,
    ) -> tuple[ControllerAction, ...]:
        """Translate one validated policy decision into side-effect-free actions."""
        if not isinstance(state, SessionState):
            raise TypeError("state must be SessionState")
        if not isinstance(policy, PolicyDecision):
            raise TypeError("policy must be PolicyDecision")

        if policy.action is PolicyAction.BACKCHANNEL:
            actions = []
            if state.response is ResponseState.DUCKED:
                actions.append(ControllerAction(ActionType.RESTORE_RESPONSE))
                state.response = ResponseState.PLAYING
            elif state.response is ResponseState.PAUSED:
                actions.append(ControllerAction(ActionType.RESUME_RESPONSE))
                state.response = ResponseState.PLAYING
            actions.append(ControllerAction(ActionType.CONTINUE_GENERATION))
            state.floor = FloorState.ASSISTANT
            return tuple(actions)

        if policy.action is PolicyAction.ANSWER:
            state.floor = FloorState.USER
            state.response = ResponseState.IDLE
            return (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.PROCESS_USER_REQUEST),
            )

        if policy.action is PolicyAction.PAUSE:
            state.floor = FloorState.USER
            state.response = ResponseState.PAUSED
            return (ControllerAction(ActionType.PAUSE_RESPONSE),)

        if policy.action is PolicyAction.RESUME:
            state.floor = FloorState.ASSISTANT
            state.response = ResponseState.PLAYING
            return (ControllerAction(ActionType.RESUME_RESPONSE),)

        if policy.action is PolicyAction.REVISE:
            state.floor = FloorState.USER
            state.response = ResponseState.CANCELLING
            return (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
                ControllerAction(ActionType.REVISE_RESPONSE),
                ControllerAction(ActionType.PROCESS_USER_REQUEST),
            )

        if policy.action is PolicyAction.NEW_REQUEST:
            state.floor = FloorState.USER
            state.response = ResponseState.CANCELLING
            return (
                ControllerAction(ActionType.STOP_RESPONSE),
                ControllerAction(ActionType.CANCEL_GENERATION),
                ControllerAction(ActionType.PROCESS_USER_REQUEST),
            )

        if policy.action is PolicyAction.MODE_SWITCH:
            if policy.intent == "chat":
                target_mode = ConversationMode.CHAT
            elif policy.intent == "continuous_interpretation":
                target_mode = ConversationMode.INTERPRETATION
            else:
                raise ValueError("unsupported MODE_SWITCH intent")
            state.mode = target_mode
            return (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": target_mode.value,
                        "source_language": policy.source_language,
                        "target_language": policy.target_language,
                    },
                ),
            )

        state.floor = FloorState.USER
        state.response = ResponseState.PAUSED
        return (
            ControllerAction(ActionType.PAUSE_RESPONSE),
            ControllerAction(ActionType.REQUEST_CLARIFICATION),
        )
