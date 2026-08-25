"""Realtime playback coordination around browser control commands and ACKs."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from src.controller.actions import ActionType, ControllerAction

from .checkpoint import ResponseCheckpointStore, ResumePlan


CommandSender = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class PlaybackAck:
    """Browser playback progress for one response segment."""

    response_id: str
    generation_epoch: int
    segment_id: int
    sample_offset: int
    audio_time: float


class PlaybackCoordinator:
    """Send playback commands and reject stale progress after epoch changes."""

    def __init__(
        self,
        *,
        send_command: CommandSender | None = None,
        checkpoint_store: ResponseCheckpointStore | None = None,
    ) -> None:
        self._send_command = send_command or (lambda _command: None)
        self.checkpoint_store = checkpoint_store or ResponseCheckpointStore()
        self._active_response_id: str | None = None
        self._active_generation_epoch: int | None = None

    def set_checkpoint_store(self, checkpoint_store: ResponseCheckpointStore) -> None:
        self.checkpoint_store = checkpoint_store

    def set_active_response(self, response_id: str, *, generation_epoch: int) -> None:
        self._active_response_id = response_id
        self._active_generation_epoch = generation_epoch

    async def apply(self, action: ControllerAction) -> ResumePlan | None:
        if action.action_type is ActionType.DUCK_RESPONSE:
            await self._emit("DUCK")
            return None
        if action.action_type is ActionType.RESTORE_RESPONSE:
            await self._emit("RESTORE")
            return None
        if action.action_type is ActionType.PAUSE_RESPONSE:
            if self._active_response_id is not None:
                self.checkpoint_store.pause(self._active_response_id)
            await self._emit("PAUSE")
            return None
        if action.action_type is ActionType.RESUME_RESPONSE:
            plan = (
                self.checkpoint_store.resume(self._active_response_id)
                if self._active_response_id is not None
                else None
            )
            await self._emit("RESUME")
            return plan
        if action.action_type is ActionType.STOP_RESPONSE:
            await self._emit("STOP")
            return None
        return None

    def ack(self, event: PlaybackAck) -> bool:
        if event.response_id != self._active_response_id:
            return False
        if event.generation_epoch != self._active_generation_epoch:
            return False
        self.checkpoint_store.ack(
            event.response_id,
            segment_id=event.segment_id,
            sample_offset=event.sample_offset,
        )
        return True

    async def _emit(self, command: str) -> None:
        result = self._send_command(command)
        if inspect.isawaitable(result):
            await result


__all__ = ["PlaybackAck", "PlaybackCoordinator"]
