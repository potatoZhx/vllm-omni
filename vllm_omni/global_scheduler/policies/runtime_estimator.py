from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.global_scheduler.types import RequestMeta


@dataclass(slots=True)
class RuntimeEstimator:
    profiling_data: dict[tuple[int | None, int | None, int | None], float] | None = None

    def estimate_runtime_s(self, request: RequestMeta, ewma_fallback_s: float) -> float:
        if self.profiling_data is None:
            return ewma_fallback_s

        key = (request.width, request.height, request.num_inference_steps)
        profiled = self.profiling_data.get(key)
        if profiled is None:
            return ewma_fallback_s
        return profiled
