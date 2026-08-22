"""Provider-neutral factories for the streaming model adapter boundaries.

The factory never imports a provider SDK and never loads a model.  A concrete
provider object is injected by the application or test harness; this module
only validates profile metadata and wraps that object in the stable adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.adapters.asr.backend import StreamingASRBackend
from src.adapters.asr.providers.whisper import WhisperASRProvider
from src.adapters.llm.backend import StreamingLLMBackend
from src.adapters.llm.providers.llama_cpp import LlamaCppLLMProvider
from src.adapters.tts.backend import StreamingTTSBackend

from .resolver import ModelProfile


class ProviderFactory:
    """Create model adapters without owning inference or lifecycle."""

    def create_asr_adapter(self, config: Any, provider: Any | None = None) -> StreamingASRBackend:
        profile = _profile(config, provider)
        if profile.provider not in {"whisper", "fake", "local", "huggingface", "modelscope"}:
            raise ValueError(f"unsupported ASR provider: {profile.provider}")
        provider_instance = profile.provider_instance
        if profile.provider == "whisper" and not isinstance(provider_instance, WhisperASRProvider):
            provider_instance = WhisperASRProvider(
                profile.model_path,
                device=profile.device or "cpu",
                options=profile.options,
                runtime=provider_instance,
            )
        return StreamingASRBackend(provider_instance, model_path=profile.model_path)

    def create_llm_adapter(self, config: Any, provider: Any | None = None) -> StreamingLLMBackend:
        profile = _profile(config, provider)
        if profile.provider not in {"llama_cpp", "fake", "local", "huggingface", "modelscope", "transformers", "vllm"}:
            raise ValueError(f"unsupported LLM provider: {profile.provider}")
        provider_instance = profile.provider_instance
        if profile.provider == "llama_cpp" and not isinstance(provider_instance, LlamaCppLLMProvider):
            provider_instance = LlamaCppLLMProvider(
                profile.model_path,
                device=profile.device or "cpu",
                options=profile.options,
                runtime=provider_instance,
            )
        return StreamingLLMBackend(provider_instance, model_path=profile.model_path)

    def create_tts_adapter(self, config: Any, provider: Any | None = None) -> StreamingTTSBackend:
        profile = _profile(config, provider)
        return StreamingTTSBackend(profile.provider_instance, model_path=profile.model_path)


def _profile(config: Any, provider: Any | None) -> "_FactoryProfile":
    if isinstance(config, ModelProfile):
        provider_name = config.provider
        model_path = config.local_path
        configured_provider = None
    elif isinstance(config, Mapping):
        provider_name = config.get("provider")
        model_path = config.get("model_path", config.get("local_path"))
        configured_provider = config.get("provider_instance", config.get("provider_runtime"))
        device = config.get("device")
        options = config.get("options")
    else:
        provider_name = getattr(config, "provider", None)
        model_path = getattr(config, "model_path", getattr(config, "local_path", None))
        configured_provider = getattr(config, "provider_instance", getattr(config, "provider_runtime", None))
        device = getattr(config, "device", None)
        options = getattr(config, "options", None)

    if isinstance(config, ModelProfile):
        device = config.device
        options = None

    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("model profile must define a provider")
    if model_path is None:
        raise ValueError("model profile must define model_path or local_path")
    provider_instance = provider if provider is not None else configured_provider
    if provider_instance is None:
        raise ValueError(
            f"provider instance is required for {provider_name!r}; model loading is outside the factory"
        )
    return _FactoryProfile(provider_name, str(Path(model_path)), provider_instance, device, options)


class _FactoryProfile:
    def __init__(self, provider: str, model_path: str, provider_instance: Any, device: str | None, options: Any) -> None:
        self.provider = provider
        self.model_path = model_path
        self.provider_instance = provider_instance
        self.device = device
        self.options = options


_FACTORY = ProviderFactory()


def create_asr_adapter(config: Any, provider: Any | None = None) -> StreamingASRBackend:
    return _FACTORY.create_asr_adapter(config, provider)


def create_llm_adapter(config: Any, provider: Any | None = None) -> StreamingLLMBackend:
    return _FACTORY.create_llm_adapter(config, provider)


def create_tts_adapter(config: Any, provider: Any | None = None) -> StreamingTTSBackend:
    return _FACTORY.create_tts_adapter(config, provider)
