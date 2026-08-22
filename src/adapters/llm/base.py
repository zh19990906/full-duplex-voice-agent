"""Model-independent streaming LLM adapter contract."""

from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """Define generation, token streaming, and cancellation boundaries."""

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
        """Cancel the active response generation."""
        raise NotImplementedError
