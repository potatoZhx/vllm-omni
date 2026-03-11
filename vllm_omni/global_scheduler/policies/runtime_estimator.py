from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from vllm_omni.global_scheduler.types import RequestMeta


@dataclass(slots=True)
class RuntimeEstimator:
    """Estimate per-request runtime from profiling with EWMA fallback."""

    profiling_data: dict[tuple[str | None, int | None, int | None, int | None, int | None], float] | None = None

    def estimate_runtime_s(
        self,
        request: RequestMeta,
        ewma_fallback_s: float,
        instance_type: str | None = None,
    ) -> float:
        """Estimate runtime in seconds for a request.

        Args:
            request: Request metadata used to build profiling key.
            ewma_fallback_s: Runtime fallback when profiling key is missing.
            instance_type: Optional instance type used for per-instance-type profiles.

        Returns:
            Estimated runtime in seconds.
        """
        if self.profiling_data is None:
            return ewma_fallback_s

        key = (
            instance_type,
            request.width,
            request.height,
            request.num_frames,
            request.num_inference_steps,
        )
        profiled = self.profiling_data.get(key)
        if profiled is None:
            estimated = self._estimate_from_neighboring_steps(
                instance_type=instance_type,
                request=request,
            )
            if estimated is None:
                return ewma_fallback_s
            return estimated
        return profiled

    def _estimate_from_neighboring_steps(
        self,
        instance_type: str | None,
        request: RequestMeta,
    ) -> float | None:
        request_steps = request.num_inference_steps
        if self.profiling_data is None or request_steps is None:
            return None

        candidates: list[tuple[int, float]] = []
        for (profile_instance_type, width, height, num_frames, steps), latency_s in self.profiling_data.items():
            if profile_instance_type != instance_type:
                continue
            if width != request.width or height != request.height or num_frames != request.num_frames:
                continue
            if steps is None:
                continue
            candidates.append((steps, latency_s))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        if len(candidates) == 1:
            profiled_steps, profiled_latency_s = candidates[0]
            if profiled_steps <= 0:
                return None
            return profiled_latency_s * (request_steps / profiled_steps)

        lower = max((item for item in candidates if item[0] <= request_steps), default=None, key=lambda item: item[0])
        upper = min((item for item in candidates if item[0] >= request_steps), default=None, key=lambda item: item[0])

        if lower is not None and upper is not None:
            if lower[0] == upper[0]:
                return lower[1]
            return self._interpolate(lower, upper, request_steps)

        if lower is None:
            return self._interpolate(candidates[0], candidates[1], request_steps)

        return self._interpolate(candidates[-2], candidates[-1], request_steps)

    @staticmethod
    def _interpolate(lower: tuple[int, float], upper: tuple[int, float], target_steps: int) -> float:
        lower_steps, lower_latency_s = lower
        upper_steps, upper_latency_s = upper
        if lower_steps == upper_steps or isclose(upper_steps, lower_steps):
            return lower_latency_s

        slope = (upper_latency_s - lower_latency_s) / float(upper_steps - lower_steps)
        return lower_latency_s + slope * float(target_steps - lower_steps)
