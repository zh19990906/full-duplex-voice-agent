"""Experiment configuration model and dependency-free loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_yaml


SUPPORTED_SCENARIOS = frozenset({"conversation", "interruption", "tool_use"})


class ExperimentConfigError(ValueError):
    """Raised for malformed or unknown experiment configuration."""


class ExperimentConfig:
    """Immutable-by-convention experiment definition."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        profile: str,
        scenario: str,
        parameters: Mapping[str, Any],
        metrics: tuple[str, ...],
    ) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.profile = profile
        self.scenario = scenario
        self.parameters = dict(parameters)
        self.metrics = tuple(metrics)
        self.validate()

    def validate(self) -> None:
        if not self.experiment_id:
            raise ExperimentConfigError("experiment id must not be empty")
        if not self.name:
            raise ExperimentConfigError("experiment name must not be empty")
        if not self.profile:
            raise ExperimentConfigError("experiment profile must not be empty")
        if self.scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"unsupported experiment scenario: {self.scenario}")
        if not all(isinstance(metric, str) and metric for metric in self.metrics):
            raise ExperimentConfigError("experiment metrics must be non-empty strings")

    @classmethod
    def from_mapping(cls, experiment_id: str, value: Mapping[str, Any]) -> "ExperimentConfig":
        scenario_value = value.get("scenario", {})
        if isinstance(scenario_value, Mapping):
            scenario = str(scenario_value.get("type", ""))
            parameters = dict(scenario_value)
            parameters.pop("type", None)
        else:
            scenario = str(scenario_value)
            parameters = {}
        parameters.update(dict(value.get("parameters", {})))
        metrics_value = value.get("metrics", ())
        if isinstance(metrics_value, str):
            metrics = tuple(item.strip() for item in metrics_value.split(",") if item.strip())
        else:
            metrics = tuple(metrics_value)
        return cls(
            experiment_id=experiment_id,
            name=str(value.get("name", experiment_id)),
            description=str(value.get("description", "")),
            profile=str(value.get("profile", "dev")),
            scenario=scenario,
            parameters=parameters,
            metrics=metrics,
        )


def load_experiment_config(path: str | Path, experiment_id: str) -> ExperimentConfig:
    try:
        document = load_yaml(path)
    except ConfigError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    experiments = document.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ExperimentConfigError("configuration requires an experiments mapping")
    value = experiments.get(experiment_id)
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"unknown experiment: {experiment_id}")
    return ExperimentConfig.from_mapping(experiment_id, value)
