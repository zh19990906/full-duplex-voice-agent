"""Streaming language model adapter contract."""

from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    """Generate response text with mandatory cancellation support."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a complete response for a prompt."""
        raise NotImplementedError

    @abstractmethod
    async def stream_tokens(self, prompt: str) -> None:
        """Start streaming response tokens for a prompt."""
        raise NotImplementedError

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the active generation."""
        raise NotImplementedError
