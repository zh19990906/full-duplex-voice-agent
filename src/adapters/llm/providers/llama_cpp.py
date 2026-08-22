"""llama.cpp-shaped streaming provider wrapper without an SDK dependency."""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any

from src.llm_runtime.stream import TokenChunk


class LlamaCppLLMProvider:
    """Adapt an injected local llama.cpp-compatible runtime to ``TokenChunk``."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        options: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("llama.cpp model_path must be supplied")
        if runtime is None:
            raise RuntimeError(
                "llama.cpp runtime is unavailable; configure a local runtime "
                "and inject it into LlamaCppLLMProvider"
            )
        self.model_path = model_path
        self.device = device
        self.options = dict(options or {})
        self.runtime = runtime
        self._cancelled = False

    async def stream_tokens(self, prompt: str) -> AsyncIterable[TokenChunk]:
        if self._cancelled:
            raise RuntimeError("llama.cpp provider has been cancelled")
        method = self._runtime_method()
        result = method(prompt, **self.options)
        if inspect.isawaitable(result):
            result = await result

        async def normalized() -> AsyncIterable[TokenChunk]:
            async for item in self._iterate(result):
                if self._cancelled:
                    return
                yield self._normalize(item)

        return normalized()

    async def generate(self, prompt: str) -> str:
        """Collect the generic token stream for the complete-response contract."""

        stream = await self.stream_tokens(prompt)
        return "".join(chunk.text async for chunk in stream)

    async def cancel(self) -> None:
        self._cancelled = True
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        self._cancelled = False

    def _runtime_method(self):
        for name in ("stream_tokens", "generate_stream"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        raise RuntimeError("llama.cpp runtime must expose stream_tokens() or generate_stream()")

    @staticmethod
    async def _iterate(result: Any) -> AsyncIterable[Any]:
        if result is None:
            return
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
            return
        if isinstance(result, (list, tuple)):
            for item in result:
                yield item
            return
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
            for item in result:
                yield item
            return
        yield result

    @staticmethod
    def _normalize(result: Any) -> TokenChunk:
        if isinstance(result, TokenChunk):
            return result
        if isinstance(result, str):
            return TokenChunk("llama-cpp-token", result, time.time(), False)
        if isinstance(result, Mapping):
            text = result.get("text", result.get("token", ""))
            if not isinstance(text, str):
                raise TypeError("llama.cpp token text must be a string")
            return TokenChunk(
                str(result.get("chunk_id", "llama-cpp-token")),
                text,
                float(result.get("timestamp", time.time())),
                bool(result.get("is_final", False)),
            )
        raise TypeError("llama.cpp result must be TokenChunk, text, mapping, or None")
