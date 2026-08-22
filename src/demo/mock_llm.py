"""Fake streaming language model adapter."""

import asyncio

from src.core.interfaces.llm import LLMAdapter


class MockLLMAdapter(LLMAdapter):
    """Stream a fixed response and stop when cancellation is requested."""

    def __init__(self, response: str = "北京旅游有三个主要景点", token_delay: float = 0.01):
        self.response = response
        self.token_delay = token_delay
        self.is_generating = False
        self.cancelled = False
        self.streamed_tokens: list[str] = []
        self.last_output = ""

    async def generate(self, prompt: str) -> str:
        """Generate the fixed response through the streaming path."""
        self.cancelled = False
        self.streamed_tokens = []
        self.last_output = ""
        await self.stream_tokens(prompt)
        return self.last_output

    async def stream_tokens(self, prompt: str) -> None:
        """Append one simulated token at a time until complete or cancelled."""
        self.is_generating = True
        try:
            for token in self.response:
                if self.cancelled:
                    break
                await asyncio.sleep(self.token_delay)
                if self.cancelled:
                    break
                self.streamed_tokens.append(token)
                self.last_output += token
        finally:
            self.is_generating = False

    async def cancel(self) -> None:
        """Request immediate cessation of the simulated generation."""
        self.cancelled = True
        self.is_generating = False
