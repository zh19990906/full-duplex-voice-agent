"""Execute controller action descriptions at the application boundary."""

from collections.abc import Iterable

from src.controller.actions import ControllerAction, ActionType
from src.generation.manager import GenerationManager


class ActionExecutor:
    """Record actions and execute only generation cancellation."""

    def __init__(self, generation_manager: GenerationManager) -> None:
        self.generation_manager = generation_manager
        self.history: list[ControllerAction] = []

    async def execute(self, actions: Iterable[ControllerAction]) -> None:
        """Record each action and apply its explicitly supported side effect."""
        for action in actions:
            self.history.append(action)
            if action.action_type is ActionType.CANCEL_GENERATION:
                await self.generation_manager.cancel_current()
