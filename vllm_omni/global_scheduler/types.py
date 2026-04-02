from dataclasses import dataclass, field
from typing import Any


SUPPORTED_BACKENDS = ("vllm-omni", "openai", "v1/videos")


@dataclass(slots=True)
class RequestMeta:
    """Scheduler-visible metadata extracted from one incoming request."""

    request_id: str
    weight: float = 1.0
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    estimated_cost_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstanceSpec:
    """Static instance specification loaded from config."""

    id: str
    endpoint: str
    instance_type: str | None = None
    numa_node: int | None = None
    launch_executable: str | None = None
    launch_model: str | None = None
    launch_args: list[str] = field(default_factory=list)
    launch_env: dict[str, str] = field(default_factory=dict)
    stop_executable: str | None = None
    stop_args: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeStats:
    """Per-instance runtime bookkeeping for policy scoring."""

    queue_len: int = 0
    inflight: int = 0
    ewma_service_time_s: float = 1.0
    outstanding_runtime_s: float = 0.0
    waiting_requests: tuple[RequestMeta, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RouteDecision:
    """Routing decision produced by the selected policy."""

    instance_id: str
    endpoint: str
    reason: str
    score: float
