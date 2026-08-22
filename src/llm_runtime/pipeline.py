"""Model-independent streaming LLM generation pipeline."""

import inspect
import time
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any
from uuid import uuid4

from src.adapters.llm.base import BaseLLMAdapter
from src.generation.manager import GenerationManager

from .session import GenerationSession, GenerationSessionStatus
from .stream import TokenChunk


class LLMGenerationPipeline:
    """Connect prompts to an LLM adapter and GenerationManager lifecycle."""

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        generation_manager: GenerationManager | None = None,
    ) -> None:
        self.adapter = adapter
        self.generation_manager = generation_manager or GenerationManager()
        self.session: GenerationSession | None = None
        self._manager_session_id: str | None = None
        self._prompt = ""
        self._chunk_number = 0

    async def start_generation(
        self,
        session_id: str | None = None,
        prompt: str = "",
    ) -> GenerationSession:
        """Create a running session and activate the shared manager session."""
        if self.session is not None and self.session.status is GenerationSessionStatus.RUNNING:
            await self.cancel_generation()

        session_id = session_id or uuid4().hex
        self.session = GenerationSession(
            session_id=session_id,
            status=GenerationSessionStatus.RUNNING,
        )
        manager_session = await self.generation_manager.start_generation(session_id)
        self._manager_session_id = manager_session.id
        self._prompt = prompt
        self._chunk_number = 0
        return self.session

    async def stream_tokens(self) -> tuple[TokenChunk, ...]:
        """Collect and return token chunks while respecting cancellation."""
        self._require_running_session()
        result = self.adapter.stream_tokens(self._prompt)
        if inspect.isawaitable(result):
            result = await result

        chunks: list[TokenChunk] = []
        if hasattr(result, "__aiter__"):
            async for item in result:
                if not self._is_running():
                    break
                chunks.append(self._normalize_token(item))
        else:
            for item in self._iter_result(result):
                if not self._is_running():
                    break
                chunks.append(self._normalize_token(item))
        return tuple(chunks)

    async def cancel_generation(self) -> GenerationSession:
        """Cancel adapter output and the matching GenerationManager session."""
        session = self._require_running_session()
        await self.adapter.cancel()
        session.status = GenerationSessionStatus.CANCELLED
        await self.generation_manager.cancel_current()
        return session

    async def complete_generation(self) -> GenerationSession:
        """Complete the active generation and clear it from the manager."""
        session = self._require_running_session()
        if self._manager_session_id is None:
            raise RuntimeError("generation manager session is not available")
        await self.generation_manager.complete(self._manager_session_id)
        session.status = GenerationSessionStatus.COMPLETED
        return session

    def _require_running_session(self) -> GenerationSession:
        if self.session is None:
            raise RuntimeError("generation session has not been started")
        if self.session.status is not GenerationSessionStatus.RUNNING:
            raise RuntimeError("generation session is not running")
        return self.session

    def _is_running(self) -> bool:
        return self.session is not None and self.session.status is GenerationSessionStatus.RUNNING

    @staticmethod
    def _iter_result(result: Any) -> Iterable[Any]:
        if result is None:
            return ()
        if isinstance(result, (str, TokenChunk, Mapping)):
            return (result,)
        if isinstance(result, Iterable):
            return result
        raise TypeError("LLM adapter result must be iterable, text, mapping, chunk, or None")

    def _normalize_token(self, result: Any) -> TokenChunk:
        if isinstance(result, TokenChunk):
            return result
        if isinstance(result, str):
            chunk = TokenChunk(
                chunk_id=f"{self.session.session_id}:{self._chunk_number}",
                text=result,
                timestamp=time.time(),
                is_final=False,
            )
        elif isinstance(result, Mapping):
            text = result.get("text", "")
            if not isinstance(text, str):
                raise TypeError("LLM result text must be a string")
            chunk = TokenChunk(
                chunk_id=str(result.get("chunk_id", f"{self.session.session_id}:{self._chunk_number}")),
                text=text,
                timestamp=float(result.get("timestamp", time.time())),
                is_final=bool(result.get("is_final", False)),
            )
        else:
            raise TypeError("LLM token must be text, mapping, or TokenChunk")
        self._chunk_number += 1
        return chunk


StreamingLLMPipeline = LLMGenerationPipeline
LLMPipeline = LLMGenerationPipeline
