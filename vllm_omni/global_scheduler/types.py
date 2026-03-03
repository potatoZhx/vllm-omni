from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestMeta:
    request_id: str
    weight: float = 1.0
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstanceSpec:
    id: str
    endpoint: str
    sp_size: int = 1
    max_concurrency: int = 1


@dataclass(slots=True)
class RuntimeStats:
    queue_len: int = 0
    inflight: int = 0
    ewma_service_time_s: float = 1.0


@dataclass(slots=True)
class RouteDecision:
    instance_id: str
    endpoint: str
    reason: str
    score: float
