from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestMeta:
    """Metadata extracted from an incoming routed request.

    Attributes:
        request_id: Stable request identifier used by logs and tracing.
        weight: Relative request weight for future weighted policies.
        width: Requested output width if available.
        height: Requested output height if available.
        num_frames: Requested frame count for video/image generation.
        num_inference_steps: Requested denoising or inference step count.
        extra: Additional scheduler-facing metadata.
    """

    request_id: str
    weight: float = 1.0
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstanceSpec:
    """Static instance specification loaded from scheduler config."""

    id: str
    endpoint: str
    sp_size: int = 1
    max_concurrency: int = 1


@dataclass(slots=True)
class RuntimeStats:
    """Mutable runtime counters maintained per upstream instance."""

    queue_len: int = 0
    inflight: int = 0
    ewma_service_time_s: float = 1.0


@dataclass(slots=True)
class RouteDecision:
    """Result of policy selection for one request."""

    instance_id: str
    endpoint: str
    reason: str
    score: float
