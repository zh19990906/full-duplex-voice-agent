"""Model-independent incremental translation adapter contract."""

from abc import ABC, abstractmethod


class BaseTranslationAdapter(ABC):
    """Define the incremental translation boundary."""

    @abstractmethod
    async def translate_stream(self, text: str) -> None:
        """Start incremental translation for text."""
        raise NotImplementedError
