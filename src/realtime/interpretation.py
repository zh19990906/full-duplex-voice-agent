"""Persistent interpretation mode orchestration for stable source segments."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import inspect
from typing import Any, Protocol

from src.asr.stream import TranscriptChunk
from src.core.interfaces.translation import TranslationAdapter

from .cancellation import CancellationToken
from .identifiers import GenerationClock, IdentifierAllocator
from .stable_prefix import StablePrefixCommitter


class TranslationSegmentSink(Protocol):
    """Publish accepted interpretation output into the response/TTS/playback path."""

    def publish(self, segment: "TranslationSegment") -> Awaitable[None] | None:
        """Accept one interpretation translation segment."""


@dataclass(frozen=True)
class SourceSegment:
    """One stable source-language segment committed exactly once."""

    source_segment_id: int
    text: str
    target_language: str
    response_id: str
    generation_epoch: int
    source_language: str | None = None
    start_offset: int = 0
    end_offset: int = 0

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
        for name in ("start_offset", "end_offset"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not be earlier than start_offset")


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
    kind: str = "TRANSLATION"
    supersedes_translation_segment_ids: tuple[int, ...] = ()

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
        if self.kind not in {"TRANSLATION", "REPLACEMENT", "CORRECTION"}:
            raise ValueError("kind must be TRANSLATION, REPLACEMENT, or CORRECTION")
        if not isinstance(self.supersedes_translation_segment_ids, tuple):
            raise TypeError("supersedes_translation_segment_ids must be a tuple")
        for value in self.supersedes_translation_segment_ids:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError("superseded translation ids must be nonnegative integers")


@dataclass
class _ActiveSource:
    source_segment: SourceSegment
    translation_segment_id: int


@dataclass
class _PendingPublication:
    signature: tuple[Any, ...]
    source_segment: SourceSegment
    translation_segment: TranslationSegment
    affected_index: int | None = None


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
    _active_sources: list[_ActiveSource] = field(default_factory=list, init=False, repr=False)
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

    @property
    def committed_source_text(self) -> str:
        return "".join(item.source_segment.text for item in self._active_sources)

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

    def preview_source_segment(
        self,
        text: str,
        *,
        start_offset: int | None = None,
    ) -> SourceSegment:
        self._activate_pending_language()
        offset = len(self.committed_source_text) if start_offset is None else start_offset
        return SourceSegment(
            source_segment_id=self._next_source_segment_id,
            text=text,
            target_language=self._active_target_language or self.target_language,
            response_id=self.response_id,
            generation_epoch=self.generation_epoch,
            source_language=self._active_source_language,
            start_offset=offset,
            end_offset=offset + len(text),
        )

    def preview_translation_segment(
        self,
        source: SourceSegment,
        translated_text: str,
        *,
        kind: str = "TRANSLATION",
        supersedes_translation_segment_ids: tuple[int, ...] = (),
    ) -> TranslationSegment:
        return TranslationSegment(
            translation_segment_id=self._next_translation_segment_id,
            source_segment_id=source.source_segment_id,
            source_text=source.text,
            translated_text=translated_text,
            target_language=source.target_language,
            response_id=source.response_id,
            generation_epoch=source.generation_epoch,
            source_language=source.source_language,
            kind=kind,
            supersedes_translation_segment_ids=supersedes_translation_segment_ids,
        )

    def commit_translation(
        self,
        source: SourceSegment,
        translation: TranslationSegment,
        *,
        affected_index: int | None = None,
        activate_source: bool = True,
    ) -> None:
        self._next_source_segment_id += 1
        self._next_translation_segment_id += 1
        self.source_segments.append(source)
        self.translation_segments.append(translation)
        if not activate_source:
            if affected_index is None:
                return
            self._active_sources = self._active_sources[:affected_index]
            return
        active = _ActiveSource(source, translation.translation_segment_id)
        if affected_index is None:
            self._active_sources.append(active)
        else:
            self._active_sources = self._active_sources[:affected_index] + [active]

    def translation_by_id(self, translation_segment_id: int) -> TranslationSegment:
        for segment in self.translation_segments:
            if segment.translation_segment_id == translation_segment_id:
                return segment
        raise KeyError(f"unknown translation_segment_id: {translation_segment_id}")

    def find_correction_target(self, corrected_text: str) -> tuple[int | None, int, tuple[int, ...]]:
        current = self.committed_source_text
        if corrected_text == current:
            return None, len(current), ()
        prefix = _longest_common_prefix(current, corrected_text)
        for index, active in enumerate(self._active_sources):
            if active.source_segment.end_offset > len(prefix):
                boundary = active.source_segment.start_offset
                affected = tuple(
                    item.translation_segment_id for item in self._active_sources[index:]
                )
                return index, boundary, affected
        boundary = len(current)
        return len(self._active_sources), boundary, ()

    def mark_translation_started(self, translation_segment_id: int) -> None:
        self.translation_by_id(translation_segment_id).playback_started = True

    def truncate_committed_state(self, affected_index: int | None) -> None:
        if affected_index is None:
            return
        self._active_sources = self._active_sources[:affected_index]

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
        sink: TranslationSegmentSink,
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
        self._lock = asyncio.Lock()
        self._accepted_results: dict[tuple[Any, ...], TranslationSegment | None] = {}
        self._pending_publication: _PendingPublication | None = None
        self._active_operation_task: asyncio.Task[Any] | None = None

    def cancel(self) -> None:
        """Cancel the active interpretation operation, if any."""
        if self.cancellation_token is not None:
            self.cancellation_token.cancel()
        task = self._active_operation_task
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()

    async def push_partial(
        self,
        hypothesis: str,
        *,
        is_final: bool = False,
    ) -> TranslationSegment | None:
        if not isinstance(hypothesis, str):
            raise TypeError("hypothesis must be a string")
        async with self._lock:
            task = asyncio.current_task()
            self._active_operation_task = task
            try:
                await self._retry_pending_locked()
                if not self._is_current():
                    return None
                stable_text = (
                    self._committer.finalize(hypothesis)
                    if is_final
                    else self._committer.update(hypothesis)
                )
                if not stable_text or not self._is_current():
                    return None
                return await self._publish_addition_locked(stable_text)
            except asyncio.CancelledError:
                if not self._is_current():
                    return None
                raise
            finally:
                if self._active_operation_task is task:
                    self._active_operation_task = None

    async def push_chunk(self, chunk: TranscriptChunk) -> TranslationSegment | None:
        if not isinstance(chunk, TranscriptChunk):
            raise TypeError("chunk must be TranscriptChunk")
        signature = (
            chunk.chunk_id,
            chunk.revision_id,
            chunk.text,
            chunk.committed_text,
            chunk.replaces_committed,
            chunk.is_final,
        )
        async with self._lock:
            task = asyncio.current_task()
            self._active_operation_task = task
            try:
                await self._retry_pending_locked()
                if signature in self._accepted_results:
                    return self._accepted_results[signature]
                if not self._is_current():
                    return None
                if chunk.replaces_committed:
                    return await self._publish_correction_locked(chunk, signature)
                if not chunk.text:
                    self._accepted_results[signature] = None
                    return None
                return await self._publish_addition_locked(chunk.text, signature=signature)
            except asyncio.CancelledError:
                if not self._is_current():
                    return None
                raise
            finally:
                if self._active_operation_task is task:
                    self._active_operation_task = None

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

    async def _publish_addition_locked(
        self,
        text: str,
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> TranslationSegment | None:
        if not self._is_current():
            return None
        source_segment = self.session.preview_source_segment(text)
        prompt = build_translation_prompt(
            source_segment.text,
            target_language=source_segment.target_language,
            source_language=source_segment.source_language,
        )
        translated_text = _translated_text(await self.translator.translate_stream(prompt))
        if not self._is_current():
            return None
        translation_segment = self.session.preview_translation_segment(source_segment, translated_text)
        return await self._publish_transactionally_locked(
            signature,
            source_segment,
            translation_segment,
        )

    async def _publish_correction_locked(
        self,
        chunk: TranscriptChunk,
        signature: tuple[Any, ...],
    ) -> TranslationSegment | None:
        corrected_text = chunk.committed_text or ""
        affected_index, boundary, superseded = self.session.find_correction_target(corrected_text)
        if not superseded and corrected_text == self.session.committed_source_text:
            self._accepted_results[signature] = None
            return None
        any_started = any(
            self.session.translation_by_id(segment_id).playback_started
            for segment_id in superseded
        )
        replacement_text = corrected_text[boundary:]
        if not replacement_text:
            if not any_started:
                self.session.truncate_committed_state(affected_index)
                self._accepted_results[signature] = None
                return None
            removed_text = self.session.committed_source_text[boundary:]
            source_segment = self.session.preview_source_segment(
                removed_text,
                start_offset=boundary,
            )
            prompt = build_discard_correction_prompt(
                source_segment.text,
                target_language=source_segment.target_language,
                source_language=source_segment.source_language,
            )
            translated_text = _translated_text(await self.translator.translate_stream(prompt))
            if not self._is_current():
                return None
            translation_segment = self.session.preview_translation_segment(
                source_segment,
                translated_text,
                kind="CORRECTION",
                supersedes_translation_segment_ids=superseded,
            )
            return await self._publish_transactionally_locked(
                signature,
                source_segment,
                translation_segment,
                affected_index=affected_index,
                activate_source=False,
            )
        source_segment = self.session.preview_source_segment(
            replacement_text,
            start_offset=boundary,
        )
        if any_started:
            prompt = build_correction_prompt(
                source_segment.text,
                target_language=source_segment.target_language,
                source_language=source_segment.source_language,
            )
            kind = "CORRECTION"
        else:
            prompt = build_translation_prompt(
                source_segment.text,
                target_language=source_segment.target_language,
                source_language=source_segment.source_language,
            )
            kind = "REPLACEMENT"
        translated_text = _translated_text(await self.translator.translate_stream(prompt))
        if not self._is_current():
            return None
        translation_segment = self.session.preview_translation_segment(
            source_segment,
            translated_text,
            kind=kind,
            supersedes_translation_segment_ids=superseded,
        )
        return await self._publish_transactionally_locked(
            signature,
            source_segment,
            translation_segment,
            affected_index=affected_index,
        )

    async def _publish_transactionally_locked(
        self,
        signature: tuple[Any, ...] | None,
        source_segment: SourceSegment,
        translation_segment: TranslationSegment,
        *,
        affected_index: int | None = None,
        activate_source: bool = True,
    ) -> TranslationSegment | None:
        if not self._is_current():
            return None
        pending = _PendingPublication(
            signature=signature or (),
            source_segment=source_segment,
            translation_segment=translation_segment,
            affected_index=affected_index,
        )
        try:
            await _publish(self.sink, translation_segment)
        except asyncio.CancelledError:
            if not self._is_current():
                return None
            raise
        except BaseException:
            self._pending_publication = pending
            raise
        if not self._is_current():
            return None
        self._commit_publication(pending, activate_source=activate_source)
        return translation_segment

    async def _retry_pending_locked(self) -> None:
        pending = self._pending_publication
        if pending is None:
            return
        if not self._is_current():
            self._pending_publication = None
            return
        try:
            await _publish(self.sink, pending.translation_segment)
        except asyncio.CancelledError:
            if not self._is_current():
                self._pending_publication = None
                return
            raise
        except BaseException:
            raise
        if not self._is_current():
            self._pending_publication = None
            return
        self._commit_publication(pending)

    def _commit_publication(
        self,
        pending: _PendingPublication,
        *,
        activate_source: bool = True,
    ) -> None:
        self.session.commit_translation(
            pending.source_segment,
            pending.translation_segment,
            affected_index=pending.affected_index,
            activate_source=activate_source,
        )
        if pending.signature:
            self._accepted_results[pending.signature] = pending.translation_segment
        self._pending_publication = None

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


def build_correction_prompt(
    source_text: str,
    *,
    target_language: str,
    source_language: str | None = None,
) -> str:
    """Return a deterministic correction instruction for already-started audio."""
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be a nonempty string")
    _require_language(target_language, "target_language")
    if source_language is not None and not isinstance(source_language, str):
        raise TypeError("source_language must be a string or None")
    rendered_source_language = source_language.strip() if source_language else "auto-detect"
    return (
        "Task: Translate the corrected source text into an explicit correction in the target language.\n"
        f"Target language: {target_language.strip()}\n"
        f"Source language: {rendered_source_language}\n"
        "Return only the correction text.\n"
        "Corrected source text:\n"
        f"{source_text}"
    )


def build_discard_correction_prompt(
    source_text: str,
    *,
    target_language: str,
    source_language: str | None = None,
) -> str:
    """Return a deterministic correction instruction for withdrawn source content."""
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be a nonempty string")
    _require_language(target_language, "target_language")
    if source_language is not None and not isinstance(source_language, str):
        raise TypeError("source_language must be a string or None")
    rendered_source_language = source_language.strip() if source_language else "auto-detect"
    return (
        "Task: Tell the listener in the target language to disregard the previous interpretation of withdrawn source content.\n"
        f"Target language: {target_language.strip()}\n"
        f"Source language: {rendered_source_language}\n"
        "Return only the correction text.\n"
        "Withdrawn source text:\n"
        f"{source_text}"
    )


async def _publish(sink: TranslationSegmentSink, segment: TranslationSegment) -> None:
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


def _longest_common_prefix(left: str, right: str) -> str:
    index = 0
    limit = min(len(left), len(right))
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


__all__ = [
    "InterpretationPipeline",
    "InterpretationSession",
    "SourceSegment",
    "TranslationSegment",
    "TranslationSegmentSink",
    "build_translation_prompt",
    "build_correction_prompt",
    "build_discard_correction_prompt",
]
