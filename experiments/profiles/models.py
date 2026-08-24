"""Serializable experiment profile and variant models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.model_runtime.resolver import ModelProfile


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    model_profile: str
    prompt_profile: str
    memory_profile: str
    tool_profile: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, variant_id: str, value: Mapping[str, Any]) -> "ExperimentVariant":
        model = value.get("model", {})
        prompt = value.get("prompt", {})
        memory = value.get("memory", {})
        tools = value.get("tools", {})
        return cls(
            variant_id=variant_id,
            model_profile=str(model.get("profile", model.get("name", variant_id))) if isinstance(model, Mapping) else str(model),
            prompt_profile=str(prompt.get("version", prompt.get("profile", "v1"))) if isinstance(prompt, Mapping) else str(prompt),
            memory_profile=str(memory.get("strategy", memory.get("profile", "recent_20"))) if isinstance(memory, Mapping) else str(memory),
            tool_profile=str(tools.get("profile", tools.get("strategy", "disabled"))) if isinstance(tools, Mapping) else str(tools),
            parameters=dict(value.get("parameters", {})),
        )


@dataclass(frozen=True)
class ModelExperimentProfile:
    profile_id: str
    asr: ModelProfile
    llm: ModelProfile
    tts: ModelProfile


@dataclass(frozen=True)
class PromptProfile:
    prompt_id: str
    version: str
    system_prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryStrategyProfile:
    strategy_id: str
    max_messages: int

    def __post_init__(self) -> None:
        if self.max_messages <= 0:
            raise ValueError("memory max_messages must be positive")


@dataclass(frozen=True)
class ToolStrategyProfile:
    strategy_id: str
    enabled_tools: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_tools)
