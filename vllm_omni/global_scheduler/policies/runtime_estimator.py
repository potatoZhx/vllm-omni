from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.global_scheduler.types import RequestMeta


@dataclass(slots=True)
class RuntimeEstimator:
    """Estimate per-request runtime from profiling with EWMA fallback."""

    profiling_data: dict[tuple[int | None, int | None, int | None], float] | None = None

    def estimate_runtime_s(self, request: RequestMeta, ewma_fallback_s: float) -> float:
        """Estimate runtime in seconds for a request.

        Args:
            request: Request metadata used to build profiling key.
            ewma_fallback_s: Runtime fallback when profiling key is missing.

        Returns:
            Estimated runtime in seconds.
        """
        if self.profiling_data is None:
            return ewma_fallback_s

        key = (request.width, request.height, request.num_inference_steps)
        profiled = self.profiling_data.get(key)
        if profiled is None:
            return ewma_fallback_s
        return profiled
