"""Small local Qwen provider for strict semantic policy decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import inspect
from typing import Any


class QwenPolicyProvider:
    """Generate one short deterministic policy JSON response.

    Tests and managed deployments inject a runtime. Local Transformers loading
    is explicit via ``load_local=True`` and remains lazy until ``generate``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cuda",
        quantization: str = "none",
        runtime: Any | None = None,
        load_local: bool = False,
        max_new_tokens: int = 64,
    ) -> None:
        if runtime is None and not load_local:
            raise ValueError("inject runtime or set load_local=True")
        if load_local and not model_path:
            raise ValueError("model_path is required for local loading")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be between 1 and 64")
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.runtime = runtime
        self.load_local = load_local
        self.max_new_tokens = max_new_tokens
        self._load_lock = asyncio.Lock()

    async def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        runtime = await self._ensure_runtime()
        method = getattr(runtime, "generate", None)
        if not callable(method):
            raise RuntimeError("policy runtime must expose generate()")
        options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }
        if inspect.iscoroutinefunction(method):
            result = await method(prompt, **options)
            return await asyncio.to_thread(_normalize_output, result)
        result = await asyncio.to_thread(_call_runtime, method, prompt, options)
        if inspect.isawaitable(result):
            result = await result
        return await asyncio.to_thread(_normalize_output, result)

    async def _ensure_runtime(self) -> Any:
        if self.runtime is not None:
            return self.runtime
        async with self._load_lock:
            if self.runtime is None:
                self.runtime = await asyncio.to_thread(
                    _load_transformers_runtime,
                    self.model_path,
                    self.device,
                    self.quantization,
                )
        return self.runtime


def _call_runtime(method: Any, prompt: str, options: Mapping[str, Any]) -> Any:
    return method(prompt, **dict(options))


def _normalize_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        value = result.get("generated_text", result.get("text"))
        if isinstance(value, str):
            return value
    if isinstance(result, (list, tuple)) and len(result) == 1:
        return _normalize_output(result[0])
    raise TypeError("policy runtime output must contain generated text")


class _TransformersPolicyRuntime:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str, **options: Any) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only one JSON object matching the policy schema. "
                    "Never add markdown or explanatory text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
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
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generated = self.model.generate(**inputs, **options)
        new_tokens = generated[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_transformers_runtime(
    model_path: str | None,
    device: str,
    quantization: str,
) -> _TransformersPolicyRuntime:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("local Qwen policy requires torch and transformers") from exc
    model_options: dict[str, Any] = {"local_files_only": True}
    if quantization == "4bit":
        model_options["load_in_4bit"] = True
    elif quantization == "8bit":
        model_options["load_in_8bit"] = True
    elif quantization != "none":
        raise ValueError("quantization must be none, 4bit, or 8bit")
    model_options["device_map"] = "auto" if device == "cuda" else device
    model_options["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_options)
    return _TransformersPolicyRuntime(tokenizer, model)
