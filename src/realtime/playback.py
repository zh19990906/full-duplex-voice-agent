"""Realtime playback coordination around browser control commands and ACKs."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.controller.actions import ActionType, ControllerAction

from .checkpoint import ResponseCheckpointStore, ResumePlan


CommandSender = Callable[[Any], Awaitable[None] | None]


@dataclass(frozen=True)
class PlaybackAck:
    """Browser playback progress for one response segment."""

    response_id: str
    generation_epoch: int
    segment_id: int
    sample_offset: int
    audio_time: float
    playback_attempt_id: int | None = None


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
        self._active_playback_attempt_id: int | None = None
        self._next_playback_attempt_id = 0
        self._attempt_required = False

    def set_checkpoint_store(self, checkpoint_store: ResponseCheckpointStore) -> None:
        self.checkpoint_store = checkpoint_store

    @property
    def active_playback_attempt_id(self) -> int | None:
        return self._active_playback_attempt_id

    def set_active_response(self, response_id: str, *, generation_epoch: int) -> None:
        self._active_response_id = response_id
        self._active_generation_epoch = generation_epoch
        self._active_playback_attempt_id = self._next_attempt_id()
        self._attempt_required = False

    def clear_active_response(self) -> None:
        self._active_response_id = None
        self._active_generation_epoch = None
        self._active_playback_attempt_id = None
        self._attempt_required = False

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
            self._active_playback_attempt_id = self._next_attempt_id()
            self._attempt_required = True
            plan = (
                self.checkpoint_store.resume(self._active_response_id)
                if self._active_response_id is not None
                else None
            )
            if plan is not None:
                plan = ResumePlan(
                    response_id=plan.response_id,
                    generation_epoch=plan.generation_epoch,
                    segment_id=plan.segment_id,
                    sample_offset=plan.sample_offset,
                    audio=plan.audio,
                    text=plan.text,
                    playback_attempt_id=self._active_playback_attempt_id,
                )
            await self._emit(
                {
                    "event": "RESUME_RESPONSE",
                    "payload": {
                        "response_id": self._active_response_id,
                        "generation_epoch": self._active_generation_epoch,
                        "playback_attempt_id": self._active_playback_attempt_id,
                    },
                }
            )
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
        if self._active_playback_attempt_id is not None:
            if event.playback_attempt_id is not None:
                if event.playback_attempt_id != self._active_playback_attempt_id:
                    return False
            elif self._attempt_required:
                return False
        self.checkpoint_store.ack(
            event.response_id,
            segment_id=event.segment_id,
            sample_offset=event.sample_offset,
        )
        return True

    def _next_attempt_id(self) -> int:
        self._next_playback_attempt_id += 1
        return self._next_playback_attempt_id

    async def _emit(self, command: Any) -> None:
        result = self._send_command(command)
        if inspect.isawaitable(result):
            await result


__all__ = ["PlaybackAck", "PlaybackCoordinator"]
