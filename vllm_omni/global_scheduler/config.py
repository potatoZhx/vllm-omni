from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ServerConfig(BaseModel):
    """Top-level scheduler server settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8089, ge=1, le=65535)
    request_timeout_s: int = Field(default=1800, ge=1)
    instance_health_check_interval_s: float = Field(default=5.0, gt=0.0)
    instance_health_check_timeout_s: float = Field(default=1.0, gt=0.0)
    instance_health_check_interval_s: float = Field(default=5.0, gt=0.0)
    instance_health_check_timeout_s: float = Field(default=1.0, gt=0.0)


class SchedulerConfig(BaseModel):
    """Global scheduler runtime tuning parameters."""

    model_config = ConfigDict(extra="forbid")

    tie_breaker: str = "random"
    ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)

    @field_validator("tie_breaker")
    @classmethod
    def validate_tie_breaker(cls, value: str) -> str:
        if value not in {"random", "lexical"}:
            raise ValueError("scheduler.tie_breaker must be one of: random, lexical")
        return value


class BaselinePolicyConfig(BaseModel):
    """Baseline policy family configuration."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "fcfs"

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        if value not in {"fcfs", "short_queue_runtime", "estimated_completion_time"}:
            raise ValueError(
                "policy.baseline.algorithm must be one of: fcfs, short_queue_runtime, estimated_completion_time"
            )
        return value


class PolicyConfig(BaseModel):
    """Policy namespace root configuration."""

    model_config = ConfigDict(extra="forbid")

    baseline: BaselinePolicyConfig = Field(default_factory=BaselinePolicyConfig)


class LaunchConfig(BaseModel):
    """Structured launch config for `vllm serve` command generation."""

    model_config = ConfigDict(extra="forbid")

    model: str
    executable: str = "vllm"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("model", "executable")
    @classmethod
    def validate_non_empty_str(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("launch.model and launch.executable cannot be empty")
        return value

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("launch.args cannot include empty items")
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not key.strip():
                raise ValueError("launch.env keys cannot be empty")
        return value


class StopConfig(BaseModel):
    """Structured stop command config for one instance."""

    model_config = ConfigDict(extra="forbid")

    executable: str = "pkill"
    args: list[str] = Field(default_factory=list)

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stop.executable cannot be empty")
        return value

    @field_validator("args")
    @classmethod
    def validate_stop_args(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("stop.args cannot include empty items")
        return value


class InstanceConfig(BaseModel):
    """Static upstream instance configuration entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    endpoint: str
    launch: LaunchConfig | None = None
    stop: StopConfig | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instances[].id cannot be empty")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http":
            raise ValueError("instances[].endpoint must be http://host:port")
        if not parsed.hostname or parsed.port is None:
            raise ValueError("instances[].endpoint must include host and port")
        if parsed.path not in {"", "/"}:
            raise ValueError("instances[].endpoint must not include path")
        return value.rstrip("/")


class GlobalSchedulerConfig(BaseModel):
    """Validated root config object for global scheduler service."""

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    instances: list[InstanceConfig]

    @model_validator(mode="after")
    def validate_unique_instance_ids(self) -> GlobalSchedulerConfig:
        instance_ids = [instance.id for instance in self.instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("instances[].id must be globally unique")
        return self


def load_config(config_path: str | Path) -> GlobalSchedulerConfig:
    """Load and validate global scheduler YAML config.

    Args:
        config_path: Path to scheduler YAML file.

    Returns:
        Parsed and validated scheduler config model.
    """
    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc

    if payload is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping in {path}")

    try:
        return GlobalSchedulerConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid global scheduler config in {path}: {exc}") from exc
