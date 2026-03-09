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


class InstanceConfig(BaseModel):
    """Static upstream instance configuration entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    endpoint: str
    sp_size: int = 1
    max_concurrency: int = Field(default=1, ge=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instances[].id cannot be empty")
        return value

    @field_validator("sp_size")
    @classmethod
    def validate_sp_size(cls, value: int) -> int:
        if value != 1:
            raise ValueError("instances[].sp_size must be 1 in current stage")
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
