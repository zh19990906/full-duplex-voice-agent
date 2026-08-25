"""Strict semantic-policy contracts for realtime conversation control."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import math
from types import MappingProxyType
from typing import Any

from src.asr.stream import TranscriptChunk
from src.realtime.session_state import SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent


class PolicyAction(str, Enum):
    """Closed semantic action set emitted by the policy model."""

    BACKCHANNEL = "BACKCHANNEL"
    ANSWER = "ANSWER"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    REVISE = "REVISE"
    NEW_REQUEST = "NEW_REQUEST"
    MODE_SWITCH = "MODE_SWITCH"
    UNCERTAIN = "UNCERTAIN"


_DECISION_KEYS = frozenset(
    {
        "action",
        "confidence",
        "rationale",
        "intent",
        "source_language",
        "target_language",
    }
)
_REQUIRED_DECISION_KEYS = frozenset({"action", "confidence", "rationale"})


@dataclass(frozen=True)
class PolicyDecision:
    """Validated, immutable semantic decision safe for transport."""

    action: PolicyAction
    confidence: float
    rationale: str
    intent: str | None = None
    source_language: str | None = None
    target_language: str | None = None

    def __post_init__(self) -> None:
        action = self.action
        if isinstance(action, str):
            try:
                action = PolicyAction(action)
            except ValueError as exc:
                raise ValueError(f"unknown policy action: {self.action!r}") from exc
            object.__setattr__(self, "action", action)
        if not isinstance(action, PolicyAction):
            raise TypeError("action must be PolicyAction")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a finite number")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        for name in ("intent", "source_language", "target_language"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if action is PolicyAction.MODE_SWITCH:
            if self.intent not in {"chat", "continuous_interpretation"}:
                raise ValueError(
                    "MODE_SWITCH intent must be 'chat' or 'continuous_interpretation'"
                )
            if self.intent == "continuous_interpretation" and (
                self.target_language is None or not self.target_language.strip()
            ):
                raise ValueError(
                    "continuous_interpretation requires a nonempty target_language"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyDecision":
        if not isinstance(value, Mapping):
            raise TypeError("policy decision must be a mapping")
        keys = frozenset(value)
        unknown = keys - _DECISION_KEYS
        missing = _REQUIRED_DECISION_KEYS - keys
        if unknown:
            raise ValueError(f"unknown policy decision fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing policy decision fields: {sorted(missing)}")
        return cls(
            action=value["action"],
            confidence=value["confidence"],
            rationale=value["rationale"],
            intent=value.get("intent"),
            source_language=value.get("source_language"),
            target_language=value.get("target_language"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "intent": self.intent,
            "source_language": self.source_language,
            "target_language": self.target_language,
        }


@dataclass(frozen=True)
class PolicyRequest:
    """Complete orthogonal context supplied to the semantic policy model."""

    state: SessionState
    assistant_last_text: str = ""
    unplayed_text_summary: str = ""
    user_transcript: TranscriptChunk | None = None
    candidate: SpeechCandidateEvent | None = None
    task_checkpoint: Mapping[str, Any] | None = None
    source_language: str | None = None
    target_language: str | None = None
    _prompt_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, SessionState):
            raise TypeError("state must be SessionState")
        for name in ("assistant_last_text", "unplayed_text_summary"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if self.user_transcript is not None and not isinstance(
            self.user_transcript, TranscriptChunk
        ):
            raise TypeError("user_transcript must be TranscriptChunk or None")
        if self.candidate is not None and not isinstance(self.candidate, SpeechCandidateEvent):
            raise TypeError("candidate must be SpeechCandidateEvent or None")
        for name in ("source_language", "target_language"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        checkpoint = self.task_checkpoint
        if checkpoint is not None:
            if not isinstance(checkpoint, Mapping):
                raise TypeError("task_checkpoint must be a mapping or None")
            frozen = json.loads(json.dumps(dict(checkpoint), ensure_ascii=False, allow_nan=False))
            object.__setattr__(self, "task_checkpoint", MappingProxyType(frozen))
        prompt_json = json.dumps(
            self._current_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "_prompt_json", prompt_json)

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached copy of the context captured at construction."""
        return json.loads(self._prompt_json)

    def _current_mapping(self) -> dict[str, Any]:
        candidate = None
        if self.candidate is not None:
            payload = self.candidate.payload
            candidate = {
                "event": self.candidate.event,
                "event_id": self.candidate.event_id,
                "timestamp": self.candidate.timestamp,
                "source": self.candidate.source,
                "label": payload.get("label"),
                "confidence": payload.get("confidence"),
                "evidence": payload.get("evidence", payload.get("source_evidence")),
            }
        return {
            "policy_contract": {
                "allowed_actions": [item.value for item in PolicyAction],
                "strict_json_only": True,
                "required_fields": ["action", "confidence", "rationale"],
                "optional_fields": ["intent", "source_language", "target_language"],
                "mode_switch_intents": ["chat", "continuous_interpretation"],
                "continuous_interpretation_requires": ["target_language"],
                "guidance": (
                    "Classify semantic intent using conversation context. A brief acknowledgement "
                    "during an assistant statement is BACKCHANNEL, but an answer to an assistant "
                    "question is ANSWER. Use MODE_SWITCH only for an explicit persistent mode "
                    "change; one-shot translation stays an ordinary request."
                ),
            },
            "state": {
                "mode": self.state.mode.value,
                "floor": self.state.floor.value,
                "response": self.state.response.value,
                "assistant_act": self.state.assistant_act,
            },
            "assistant_last_text": self.assistant_last_text,
            "unplayed_text_summary": self.unplayed_text_summary,
            "user_transcript": (
                self.user_transcript.to_dict() if self.user_transcript is not None else None
            ),
            "candidate": candidate,
            "task_checkpoint": (
                dict(self.task_checkpoint) if self.task_checkpoint is not None else None
            ),
            "source_language": self.source_language,
            "target_language": self.target_language,
        }

    def to_prompt_json(self) -> str:
        """Return deterministic JSON without adding keyword business rules."""
        return self._prompt_json


class SemanticPolicyEngine:
    """Run an injected policy provider with a strict latency and schema boundary."""

    def __init__(
        self,
        provider: Any,
        *,
        timeout_ms: int = 300,
        confidence_threshold: float = 0.7,
        on_decision: Callable[[PolicyDecision], Any] | None = None,
    ) -> None:
        if provider is None or not callable(getattr(provider, "generate", None)):
            raise TypeError("policy provider must expose generate(prompt)")
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 300:
            raise ValueError("timeout_ms must be an integer between 1 and 300")
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(float(confidence_threshold))
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.provider = provider
        self.timeout_seconds = timeout_ms / 1000.0
        self.confidence_threshold = float(confidence_threshold)
        self.on_decision = on_decision
        self._inflight_task: asyncio.Task[Any] | None = None
        self._inflight_lock = asyncio.Lock()

    @property
    def inflight(self) -> bool:
        """Whether one provider generation is still consuming resources."""
        task = self._inflight_task
        return task is not None and not task.done()

    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        if not isinstance(request, PolicyRequest):
            raise TypeError("request must be PolicyRequest")
        task = await self._start_generation(request.to_prompt_json())
        if task is None:
            decision = _uncertain("policy busy")
            await self._publish(decision)
            return decision
        try:
            raw = await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            decision = _uncertain("policy timeout")
        except Exception:
            decision = _uncertain("policy runtime failure")
        else:
            try:
                decision = self._parse(raw)
                if decision.confidence < self.confidence_threshold:
                    decision = _uncertain("policy confidence below threshold")
            except (TypeError, ValueError, json.JSONDecodeError):
                decision = _uncertain("policy output rejected")
        await self._publish(decision)
        return decision

    async def _start_generation(self, prompt: str) -> asyncio.Task[Any] | None:
        async with self._inflight_lock:
            current = self._inflight_task
            if current is not None and not current.done():
                return None
            self._inflight_task = asyncio.create_task(self._generate(prompt))
            task = self._inflight_task
            loop = asyncio.get_running_loop()
            task.add_done_callback(
                lambda finished: _schedule_inflight_cleanup(loop, self, finished)
            )
            return task

    async def _clear_inflight(self, task: asyncio.Task[Any]) -> None:
        async with self._inflight_lock:
            if self._inflight_task is task:
                self._inflight_task = None

    async def _generate(self, prompt: str) -> Any:
        method = self.provider.generate
        if inspect.iscoroutinefunction(method):
            return await method(prompt)
        result = await asyncio.to_thread(method, prompt)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _parse(raw: Any) -> PolicyDecision:
        if not isinstance(raw, str):
            raise TypeError("policy output must be a JSON string")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise TypeError("policy JSON must contain an object")
        return PolicyDecision.from_mapping(value)

    async def _publish(self, decision: PolicyDecision) -> None:
        if self.on_decision is None:
            return
        result = self.on_decision(decision)
        if inspect.isawaitable(result):
            await result


def _uncertain(rationale: str) -> PolicyDecision:
    return PolicyDecision(PolicyAction.UNCERTAIN, 0.0, rationale)


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


def _schedule_inflight_cleanup(
    loop: asyncio.AbstractEventLoop,
    engine: SemanticPolicyEngine,
    task: asyncio.Task[Any],
) -> None:
    _consume_task_result(task)
    if loop.is_closed():
        return
    cleanup = loop.create_task(engine._clear_inflight(task))
    cleanup.add_done_callback(_consume_task_result)
