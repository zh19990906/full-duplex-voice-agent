"""Persistent interpretation mode orchestration for stable source segments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import inspect
from typing import Any

from src.core.interfaces.translation import TranslationAdapter

from .cancellation import CancellationToken
from .identifiers import GenerationClock, IdentifierAllocator
from .stable_prefix import StablePrefixCommitter


SinkPublisher = Callable[[Any], Awaitable[None] | None]


@dataclass(frozen=True)
class SourceSegment:
    """One stable source-language segment committed exactly once."""

    source_segment_id: int
    text: str
    target_language: str
    response_id: str
    generation_epoch: int
    source_language: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source_segment_id, bool) or not isinstance(self.source_segment_id, int):
            raise TypeError("source_segment_id must be an integer")
        if self.source_segment_id < 0:
            raise ValueError("source_segment_id must be nonnegative")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be a nonempty string")
        if not isinstance(self.target_language, str) or not self.target_language.strip():
            raise ValueError("target_language must be a nonempty string")
        if not isinstance(self.response_id, str) or not self.response_id:
            raise ValueError("response_id must be a nonempty string")
        if isinstance(self.generation_epoch, bool) or not isinstance(self.generation_epoch, int):
            raise TypeError("generation_epoch must be an integer")
        if self.generation_epoch < 0:
            raise ValueError("generation_epoch must be nonnegative")
        if self.source_language is not None and not isinstance(self.source_language, str):
            raise TypeError("source_language must be a string or None")


@dataclass
class TranslationSegment:
    """One immutable translation for one stable source segment."""

    translation_segment_id: int
    source_segment_id: int
    source_text: str
    translated_text: str
    target_language: str
    response_id: str
    generation_epoch: int
    source_language: str | None = None
    playback_started: bool = False

    def __post_init__(self) -> None:
        for name in ("translation_segment_id", "source_segment_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("source_text", "translated_text", "target_language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.response_id, str) or not self.response_id:
            raise ValueError("response_id must be a nonempty string")
        if isinstance(self.generation_epoch, bool) or not isinstance(self.generation_epoch, int):
            raise TypeError("generation_epoch must be an integer")
        if self.generation_epoch < 0:
            raise ValueError("generation_epoch must be nonnegative")
        if self.source_language is not None and not isinstance(self.source_language, str):
            raise TypeError("source_language must be a string or None")
        if not isinstance(self.playback_started, bool):
            raise TypeError("playback_started must be a bool")


@dataclass
class InterpretationSession:
    """Track persistent interpretation direction and monotonic segment identity."""

    target_language: str
    source_language: str | None = None
    generation_clock: GenerationClock = field(default_factory=GenerationClock)
    response_id_factory: Callable[[], str] | None = None
    _active_target_language: str | None = field(default=None, init=False, repr=False)
    _active_source_language: str | None = field(default=None, init=False, repr=False)
    _pending_target_language: str | None = field(default=None, init=False, repr=False)
    _pending_source_language: str | None = field(default=None, init=False, repr=False)
    _next_source_segment_id: int = field(default=0, init=False, repr=False)
    _next_translation_segment_id: int = field(default=0, init=False, repr=False)
    response_id: str = field(default="", init=False)
    generation_epoch: int = field(default=0, init=False)
    source_segments: list[SourceSegment] = field(default_factory=list, init=False)
    translation_segments: list[TranslationSegment] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _require_language(self.target_language, "target_language")
        if self.source_language is not None and not isinstance(self.source_language, str):
            raise TypeError("source_language must be a string or None")
        if not isinstance(self.generation_clock, GenerationClock):
            raise TypeError("generation_clock must be a GenerationClock")
        allocator = IdentifierAllocator()
        factory = self.response_id_factory or allocator.next_response_id
        response_id = factory()
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("response_id_factory must return a nonempty string")
        self.response_id = response_id
        self.generation_epoch = self.generation_clock.current
        self._pending_target_language = self.target_language.strip()
        self._pending_source_language = self.source_language

    def set_target_language(
        self,
        target_language: str,
        *,
        source_language: str | None = None,
    ) -> None:
        _require_language(target_language, "target_language")
        if source_language is not None and not isinstance(source_language, str):
            raise TypeError("source_language must be a string or None")
        self.target_language = target_language.strip()
        self.source_language = source_language
        self._pending_target_language = self.target_language
        self._pending_source_language = self.source_language

    def next_source_segment(self, text: str) -> SourceSegment:
        self._activate_pending_language()
        segment = SourceSegment(
            source_segment_id=self._next_source_segment_id,
            text=text,
            target_language=self._active_target_language or self.target_language,
            response_id=self.response_id,
            generation_epoch=self.generation_epoch,
            source_language=self._active_source_language,
        )
        self._next_source_segment_id += 1
        self.source_segments.append(segment)
        return segment

    def record_translation(self, source: SourceSegment, translated_text: str) -> TranslationSegment:
        segment = TranslationSegment(
            translation_segment_id=self._next_translation_segment_id,
            source_segment_id=source.source_segment_id,
            source_text=source.text,
            translated_text=translated_text,
            target_language=source.target_language,
            response_id=source.response_id,
            generation_epoch=source.generation_epoch,
            source_language=source.source_language,
        )
        self._next_translation_segment_id += 1
        self.translation_segments.append(segment)
        return segment

    def mark_translation_started(self, translation_segment_id: int) -> None:
        for segment in self.translation_segments:
            if segment.translation_segment_id == translation_segment_id:
                segment.playback_started = True
                return
        raise KeyError(f"unknown translation_segment_id: {translation_segment_id}")

    def _activate_pending_language(self) -> None:
        if self._pending_target_language is not None:
            self._active_target_language = self._pending_target_language
            self._active_source_language = self._pending_source_language
            self._pending_target_language = None
            self._pending_source_language = None


class InterpretationPipeline:
    """Translate only newly stable source text and publish it once."""

    def __init__(
        self,
        *,
        translator: TranslationAdapter,
        sink: Any,
        target_language: str,
        source_language: str | None = None,
        cancellation_token: CancellationToken | None = None,
        generation_clock: GenerationClock | None = None,
        response_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not callable(getattr(translator, "translate_stream", None)):
            raise TypeError("translator must expose translate_stream(text)")
        if not callable(getattr(sink, "publish", None)):
            raise TypeError("sink must expose publish(segment)")
        if cancellation_token is not None and not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken")
        self.translator = translator
        self.sink = sink
        self.cancellation_token = cancellation_token
        self.session = InterpretationSession(
            target_language=target_language,
            source_language=source_language,
            generation_clock=generation_clock or GenerationClock(),
            response_id_factory=response_id_factory,
        )
        self._committer = StablePrefixCommitter()

    async def push_partial(
        self,
        hypothesis: str,
        *,
        is_final: bool = False,
    ) -> TranslationSegment | None:
        if not isinstance(hypothesis, str):
            raise TypeError("hypothesis must be a string")
        if not self._is_current():
            return None
        stable_text = (
            self._committer.finalize(hypothesis)
            if is_final
            else self._committer.update(hypothesis)
        )
        if not stable_text or not self._is_current():
            return None
        source_segment = self.session.next_source_segment(stable_text)
        prompt = build_translation_prompt(
            source_segment.text,
            target_language=source_segment.target_language,
            source_language=source_segment.source_language,
        )
        translated_text = _translated_text(await self.translator.translate_stream(prompt))
        if not self._is_current():
            return None
        translation_segment = self.session.record_translation(source_segment, translated_text)
        await _publish(self.sink, translation_segment)
        return translation_segment

    def set_target_language(
        self,
        target_language: str,
        *,
        source_language: str | None = None,
    ) -> None:
        self.session.set_target_language(
            target_language,
            source_language=source_language,
        )

    def mark_translation_started(self, translation_segment_id: int) -> None:
        self.session.mark_translation_started(translation_segment_id)

    def _cancelled(self) -> bool:
        return self.cancellation_token is not None and self.cancellation_token.is_cancelled()

    def _is_current(self) -> bool:
        return self.session.generation_clock.is_current(self.session.generation_epoch) and not (
            self._cancelled()
        )


def build_translation_prompt(
    source_text: str,
    *,
    target_language: str,
    source_language: str | None = None,
) -> str:
    """Return a deterministic, model-independent translation instruction."""
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be a nonempty string")
    _require_language(target_language, "target_language")
    if source_language is not None and not isinstance(source_language, str):
        raise TypeError("source_language must be a string or None")
    rendered_source_language = source_language.strip() if source_language else "auto-detect"
    return (
        "Task: Translate the source text into the target language.\n"
        f"Target language: {target_language.strip()}\n"
        f"Source language: {rendered_source_language}\n"
        "Return only the translation text.\n"
        "Source text:\n"
        f"{source_text}"
    )


async def _publish(sink: Any, segment: TranslationSegment) -> None:
    result = sink.publish(segment)
    if inspect.isawaitable(result):
        await result


def _translated_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, TranslationSegment):
        return result.translated_text
    if isinstance(result, Mapping):
        value = result.get("translated_text", "")
        if not isinstance(value, str):
            raise TypeError("translation result translated_text must be a string")
        return value
    raise TypeError("translation adapter result must be text, mapping, TranslationSegment, or None")


def _require_language(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


__all__ = [
    "InterpretationPipeline",
    "InterpretationSession",
    "SourceSegment",
    "TranslationSegment",
    "build_translation_prompt",
]
