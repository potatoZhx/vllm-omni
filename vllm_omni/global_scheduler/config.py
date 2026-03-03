from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8089, ge=1, le=65535)
    request_timeout_s: int = Field(default=1800, ge=1)


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "baseline_sp1"
    tie_breaker: str = "random"
    ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in {"baseline_sp1", "ondisc_sp1"}:
            raise ValueError("scheduler.type must be one of: baseline_sp1, ondisc_sp1")
        return value

    @field_validator("tie_breaker")
    @classmethod
    def validate_tie_breaker(cls, value: str) -> str:
        if value not in {"random", "lexical"}:
            raise ValueError("scheduler.tie_breaker must be one of: random, lexical")
        return value


class BaselinePolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "ect"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in {"shortest_queue", "ect"}:
            raise ValueError("policy.baseline_sp1.mode must be one of: shortest_queue, ect")
        return value


class OnDiscPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 1.0
    delta: float = 0.0
    epsilon: float = 0.1


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_sp1: BaselinePolicyConfig = Field(default_factory=BaselinePolicyConfig)
    ondisc_sp1: OnDiscPolicyConfig = Field(default_factory=OnDiscPolicyConfig)


class InstanceConfig(BaseModel):
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
