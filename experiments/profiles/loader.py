"""Load experiment profile selections without loading models or creating agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_yaml
from src.model_runtime.resolver import ModelProfile, load_model_config

from .models import (
    ExperimentVariant,
    MemoryStrategyProfile,
    ModelExperimentProfile,
    PromptProfile,
    ToolStrategyProfile,
)


class ExperimentProfileError(ValueError):
    """Raised for an unknown or malformed experiment profile."""


@dataclass(frozen=True)
class ExperimentRuntimeContext:
    variant: ExperimentVariant
    model: ModelExperimentProfile
    prompt: PromptProfile
    memory: MemoryStrategyProfile
    tools: ToolStrategyProfile


class ExperimentProfileBundle:
    """Resolve named experiment profiles to existing deployment definitions."""

    def __init__(
        self,
        variants: Mapping[str, ExperimentVariant],
        models: Mapping[str, ModelExperimentProfile],
        prompts: Mapping[str, PromptProfile],
        memory: Mapping[str, MemoryStrategyProfile],
        tools: Mapping[str, ToolStrategyProfile],
    ) -> None:
        self.variants = dict(variants)
        self.models = dict(models)
        self.prompts = dict(prompts)
        self.memory_strategies = dict(memory)
        self.tool_strategies = dict(tools)

    @classmethod
    def from_files(
        cls,
        variants_path: str | Path,
        models_path: str | Path,
        prompts_path: str | Path,
        strategies_path: str | Path,
        deployment_path: str | Path = "configs/models.yaml",
    ) -> "ExperimentProfileBundle":
        try:
            variant_doc = load_yaml(variants_path)
            prompt_doc = load_yaml(prompts_path)
            strategy_doc = load_yaml(strategies_path)
        except ConfigError as exc:
            raise ExperimentProfileError(str(exc)) from exc
        try:
            deployment = load_model_config(deployment_path)
        except (OSError, ValueError) as exc:
            raise ExperimentProfileError(str(exc)) from exc

        variant_values = variant_doc.get("variants")
        if not isinstance(variant_values, Mapping):
            raise ExperimentProfileError("experiment profiles require a variants mapping")
        variants: dict[str, ExperimentVariant] = {}
        for name, value in variant_values.items():
            if not isinstance(value, Mapping):
                raise ExperimentProfileError(f"invalid experiment variant: {name}")
            variants[name] = ExperimentVariant.from_mapping(name, value)

        # The experiment model profile file is intentionally a mapping of aliases
        # to names in configs/models.yaml; it does not duplicate deployment data.
        model_doc = load_yaml(models_path)
        model_aliases = model_doc.get("profiles", {})
        if not isinstance(model_aliases, Mapping):
            raise ExperimentProfileError("model experiment profiles require a profiles mapping")
        models: dict[str, ModelExperimentProfile] = {}
        for profile_id, value in model_aliases.items():
            if not isinstance(value, Mapping):
                raise ExperimentProfileError(f"invalid model experiment profile: {profile_id}")
            models[profile_id] = ModelExperimentProfile(
                profile_id,
                _deployment_profile(deployment.profiles, value, "asr", profile_id),
                _deployment_profile(deployment.profiles, value, "llm", profile_id),
                _deployment_profile(deployment.profiles, value, "tts", profile_id),
            )

        prompts = {}
        prompt_values = prompt_doc.get("prompts", {})
        if not isinstance(prompt_values, Mapping):
            raise ExperimentProfileError("prompt profiles require a prompts mapping")
        for prompt_id, value in prompt_values.items():
            value = value if isinstance(value, Mapping) else {}
            metadata = value.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ExperimentProfileError(f"invalid prompt metadata: {prompt_id}")
            prompts[prompt_id] = PromptProfile(
                prompt_id=prompt_id,
                version=str(value.get("version", prompt_id)),
                system_prompt=str(value.get("system", "")),
                metadata=dict(metadata),
            )

        memory = {}
        memory_values = strategy_doc.get("memory", {})
        if not isinstance(memory_values, Mapping):
            raise ExperimentProfileError("memory strategies require a memory mapping")
        for strategy_id, value in memory_values.items():
            value = value if isinstance(value, Mapping) else {}
            memory[strategy_id] = MemoryStrategyProfile(strategy_id, int(value.get("max_messages", 20)))

        tools = {}
        tool_values = strategy_doc.get("tools", {})
        if not isinstance(tool_values, Mapping):
            raise ExperimentProfileError("tool strategies require a tools mapping")
        for strategy_id, value in tool_values.items():
            value = value if isinstance(value, Mapping) else {}
            enabled = value.get("enabled", False)
            if enabled is True:
                enabled_tools = tuple(value.get("tools", "").split(",")) if isinstance(value.get("tools"), str) else ()
            elif isinstance(enabled, str):
                enabled_tools = tuple(item.strip() for item in enabled.split(",") if item.strip())
            else:
                enabled_tools = tuple(enabled) if isinstance(enabled, (list, tuple)) else ()
            tools[strategy_id] = ToolStrategyProfile(strategy_id, enabled_tools)

        return cls(variants, models, prompts, memory, tools)

    def variant(self, variant_id: str) -> ExperimentVariant:
        return _get(self.variants, variant_id, "variant")

    def model(self, profile_id: str) -> ModelExperimentProfile:
        return _get(self.models, profile_id, "model profile")

    def prompt(self, profile_id: str) -> PromptProfile:
        return _get(self.prompts, profile_id, "prompt profile")

    def memory(self, profile_id: str) -> MemoryStrategyProfile:
        return _get(self.memory_strategies, profile_id, "memory strategy")

    def tools(self, profile_id: str) -> ToolStrategyProfile:
        return _get(self.tool_strategies, profile_id, "tool strategy")

    def context(self, variant_id: str | ExperimentVariant) -> ExperimentRuntimeContext:
        variant = variant_id if isinstance(variant_id, ExperimentVariant) else self.variant(variant_id)
        return ExperimentRuntimeContext(
            variant=variant,
            model=self.model(variant.model_profile),
            prompt=self.prompt(variant.prompt_profile),
            memory=self.memory(variant.memory_profile),
            tools=self.tools(variant.tool_profile),
        )


def _get(values: Mapping[str, Any], name: str, label: str) -> Any:
    try:
        return values[name]
    except KeyError as exc:
        raise ExperimentProfileError(f"unknown {label}: {name}") from exc


def _deployment_profile(
    profiles: Mapping[str, ModelProfile],
    aliases: Mapping[str, Any],
    component: str,
    profile_id: str,
) -> ModelProfile:
    name = aliases.get(component)
    if not isinstance(name, str) or name not in profiles:
        raise ExperimentProfileError(f"model profile {profile_id!r} references unknown {component}: {name}")
    return profiles[name]
