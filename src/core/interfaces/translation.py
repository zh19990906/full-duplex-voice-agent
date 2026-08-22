"""Incremental translation adapter contract."""

from abc import ABC, abstractmethod


class TranslationAdapter(ABC):
    """Translate an input stream incrementally."""

    @abstractmethod
    async def translate_stream(self, text: str) -> None:
        """Start incremental translation for text."""
        raise NotImplementedError
