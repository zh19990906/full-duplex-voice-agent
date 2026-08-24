"""Transformers-backed streaming provider for local Qwen chat models."""

from __future__ import annotations

import asyncio
import inspect
import queue
import time
from collections.abc import AsyncIterable, Mapping
from typing import Any

from src.llm_runtime.stream import TokenChunk


class TransformersQwenProvider:
    """Expose a local Transformers model through the project LLM contract.

    A runtime can be injected for tests or for deployments that own model
    loading. Without one, the provider lazily imports Transformers and loads a
    Qwen-compatible causal language model from ``model_path``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
        import_runtime: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("Qwen model_path must be supplied")
        self.model_path = model_path
        self.device = device
        self.options = dict(options or {})
        self.runtime = runtime
        self._cancelled = False
        self._generation_task: asyncio.Task[Any] | None = None
        if self.runtime is None and import_runtime:
            self.runtime = self._load_runtime()
        if self.runtime is None:
            raise RuntimeError(
                "Qwen Transformers runtime is unavailable; install transformers "
                "and inject or load a compatible runtime"
            )

    async def stream_tokens(self, prompt: str) -> AsyncIterable[TokenChunk]:
        """Stream normalized token chunks from the configured runtime."""

        if self._cancelled:
            raise RuntimeError("Qwen Transformers provider has been cancelled")
        method = getattr(self.runtime, "stream_tokens", None)
        if not callable(method):
            raise RuntimeError("Qwen runtime must expose stream_tokens()")
        result = method(prompt, **self.options)
        if inspect.isawaitable(result):
            result = await result
        async for item in self._iterate(result):
            if self._cancelled:
                return
            yield self._normalize(item)

    async def generate(self, prompt: str) -> str:
        """Generate a complete response using the runtime or token stream."""

        if self._cancelled:
            raise RuntimeError("Qwen Transformers provider has been cancelled")
        method = getattr(self.runtime, "generate", None)
        if callable(method):
            result = method(prompt, **self.options)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, str):
                raise TypeError("Qwen runtime generate result must be a string")
            return result
        chunks = [chunk async for chunk in self.stream_tokens(prompt)]
        return "".join(chunk.text for chunk in chunks)

    async def cancel(self) -> None:
        """Stop accepting output and notify a runtime cancellation hook."""

        self._cancelled = True
        if self._generation_task is not None:
            self._generation_task.cancel()
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            result = cancel()
            if inspect.isawaitable(result):
                await result

    def reset(self) -> None:
        """Allow a new generation after cancellation."""

        self._cancelled = False
        reset = getattr(self.runtime, "reset", None)
        if callable(reset):
            reset()

    async def _iterate(self, result: Any) -> AsyncIterable[Any]:
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
        yield result

    @staticmethod
    def _normalize(result: Any) -> TokenChunk:
        if isinstance(result, TokenChunk):
            return result
        if isinstance(result, str):
            return TokenChunk("qwen-token", result, time.time(), False)
        if isinstance(result, Mapping):
            text = result.get("text", result.get("token", ""))
            if not isinstance(text, str):
                raise TypeError("Qwen token text must be a string")
            return TokenChunk(
                str(result.get("chunk_id", "qwen-token")),
                text,
                float(result.get("timestamp", time.time())),
                bool(result.get("is_final", False)),
            )
        raise TypeError("Qwen result must be TokenChunk, text, or mapping")

    def _load_runtime(self) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen Transformers runtime requires torch and transformers"
            ) from exc

        dtype_name = str(self.options.pop("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        model_options = dict(self.options)
        max_new_tokens = int(model_options.pop("max_new_tokens", 512))
        temperature = float(model_options.pop("temperature", 0.7))
        top_p = float(model_options.pop("top_p", 0.9))

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            local_files_only=True,
        )
        return _TransformersRuntime(
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            model_options=model_options,
        )


class _TransformersRuntime:
    def __init__(self, tokenizer: Any, model: Any, **generation_options: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        nested_options = generation_options.pop("model_options", {})
        if not isinstance(nested_options, Mapping):
            raise TypeError("model_options must be a mapping")
        self.generation_options = {**dict(nested_options), **generation_options}
        self._cancelled = False

    async def stream_tokens(self, prompt: str, **options: Any) -> AsyncIterable[TokenChunk]:
        import torch
        from transformers import TextIteratorStreamer

        self._cancelled = False
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=1.0,
        )
        generation = dict(self.generation_options)
        generation.update(options)
        max_new_tokens = int(generation.pop("max_new_tokens", 512))
        temperature = float(generation.pop("temperature", 0.7))
        top_p = float(generation.pop("top_p", 0.9))
        if temperature <= 0:
            generation["do_sample"] = False
        else:
            generation.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        generation.update({"max_new_tokens": max_new_tokens, "streamer": streamer})
        generation_task = asyncio.create_task(
            asyncio.to_thread(self.model.generate, **inputs, **generation)
        )
        try:
            index = 0
            while True:
                if self._cancelled:
                    return
                item = await asyncio.to_thread(_next_or_timeout, streamer)
                if item is _STREAM_TIMEOUT:
                    if generation_task.done() and not generation_task.cancelled():
                        error = generation_task.exception()
                        if error is not None:
                            raise error
                    continue
                if item is None:
                    break
                yield TokenChunk(f"qwen-{index}", item, time.time(), False)
                index += 1
            await generation_task
            yield TokenChunk(f"qwen-{index}", "", time.time(), True)
        finally:
            if not generation_task.done():
                generation_task.cancel()

    async def generate(self, prompt: str, **options: Any) -> str:
        chunks = [chunk async for chunk in self.stream_tokens(prompt, **options)]
        return "".join(chunk.text for chunk in chunks)

    async def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False


_STREAM_TIMEOUT = object()


def _next_or_timeout(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return None
    except (queue.Empty, TimeoutError):
        return _STREAM_TIMEOUT
