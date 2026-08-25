"""Stable language-aware text segmentation for streaming speech synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


_CLAUSE_PUNCTUATION = frozenset("，。！？；：,.!?;:\n")
_ASCII_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")


@dataclass(frozen=True)
class TextSegment:
    """One nonempty stable TTS unit.

    The first three fields intentionally retain the original positional API.
    Realtime callers additionally fill ``response_id`` and ``generation_epoch``
    before the segment crosses a transport boundary.
    """

    segment_id: int
    text: str
    is_final: bool
    response_id: str | None = None
    generation_epoch: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.segment_id, bool) or not isinstance(self.segment_id, int):
            raise TypeError("segment_id must be an integer")
        if self.segment_id < 0:
            raise ValueError("segment_id must be nonnegative")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text:
            raise ValueError("text must be nonempty")
        if not isinstance(self.is_final, bool):
            raise TypeError("is_final must be a bool")
        if self.response_id is not None and (
            not isinstance(self.response_id, str) or not self.response_id
        ):
            raise ValueError("response_id must be a nonempty string or None")
        if self.generation_epoch is not None:
            if isinstance(self.generation_epoch, bool) or not isinstance(
                self.generation_epoch, int
            ):
                raise TypeError("generation_epoch must be an integer or None")
            if self.generation_epoch < 0:
                raise ValueError("generation_epoch must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        """Return the transport-safe identity-bearing representation."""
        return asdict(self)


class LanguageAwareTextSegmenter:
    """Incrementally emit punctuation-preferred, bounded TTS text segments."""

    def __init__(
        self,
        *,
        first_min_chars: int = 12,
        next_min_chars: int = 24,
        first_max_chars: int = 24,
        next_max_chars: int = 48,
        first_min_words: int = 6,
        next_min_words: int = 10,
    ) -> None:
        for name, value in {
            "first_min_chars": first_min_chars,
            "next_min_chars": next_min_chars,
            "first_max_chars": first_max_chars,
            "next_max_chars": next_max_chars,
            "first_min_words": first_min_words,
            "next_min_words": next_min_words,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.first_min_chars = first_min_chars
        self.next_min_chars = next_min_chars
        self.first_max_chars = first_max_chars
        self.next_max_chars = next_max_chars
        self.first_min_words = first_min_words
        self.next_min_words = next_min_words
        self.reset()

    def push(self, text: str) -> tuple[TextSegment, ...]:
        """Accept an arbitrary token boundary and return newly stable segments."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return ()
        self._buffer += text
        return self._drain(final=False)

    def flush(self, text: str | None = None) -> tuple[TextSegment, ...]:
        """Optionally add final text and emit the remaining buffer exactly once."""
        if text is not None:
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            self._buffer += text
        emitted = list(self._drain(final=False))
        if self._buffer:
            emitted.append(self._emit(self._buffer, is_final=True))
            self._buffer = ""
        elif emitted:
            last = emitted[-1]
            emitted[-1] = TextSegment(last.segment_id, last.text, True)
        return tuple(emitted)

    def reset(self) -> None:
        """Discard uncommitted text and start a fresh response at segment zero."""
        self._buffer = ""
        self._next_segment_id = 0

    def _drain(self, *, final: bool) -> tuple[TextSegment, ...]:
        emitted: list[TextSegment] = []
        while self._buffer:
            boundary = self._next_boundary()
            if boundary is None:
                break
            emitted.append(self._emit(self._buffer[:boundary], is_final=final))
            self._buffer = self._buffer[boundary:]
        return tuple(emitted)

    def _next_boundary(self) -> int | None:
        chinese = _is_chinese_led(self._buffer)
        minimum = self._minimum(chinese)
        maximum = self._maximum()
        for index, character in enumerate(self._buffer, start=1):
            if character in _CLAUSE_PUNCTUATION and self._meets_minimum(
                self._buffer[:index], chinese, minimum
            ):
                return index
        if not chinese:
            for index, character in enumerate(self._buffer, start=1):
                if character.isspace() and index > 1 and self._meets_minimum(
                    self._buffer[: index - 1], chinese, minimum
                ):
                    return index - 1
        if len(self._buffer) < maximum:
            return None
        if chinese:
            return maximum
        return _english_hard_boundary(self._buffer, maximum)

    def _minimum(self, chinese: bool) -> int:
        if chinese:
            return self.first_min_chars if self._next_segment_id == 0 else self.next_min_chars
        return self.first_min_words if self._next_segment_id == 0 else self.next_min_words

    def _maximum(self) -> int:
        return self.first_max_chars if self._next_segment_id == 0 else self.next_max_chars

    @staticmethod
    def _meets_minimum(text: str, chinese: bool, minimum: int) -> bool:
        return len(text) >= minimum if chinese else len(_ASCII_WORD.findall(text)) >= minimum

    def _emit(self, text: str, *, is_final: bool) -> TextSegment:
        segment = TextSegment(self._next_segment_id, text, is_final)
        self._next_segment_id += 1
        return segment


def _is_chinese_led(text: str) -> bool:
    cjk = sum("\u4e00" <= character <= "\u9fff" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    return cjk >= latin and cjk > 0


def _english_hard_boundary(text: str, maximum: int) -> int:
    prefix = text[:maximum]
    whitespace = max(prefix.rfind(" "), prefix.rfind("\t"), prefix.rfind("\n"))
    return whitespace + 1 if whitespace > 0 else maximum


__all__ = ["LanguageAwareTextSegmenter", "TextSegment"]
