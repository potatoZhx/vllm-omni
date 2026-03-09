"""Runtime estimator profiling-hit and fallback tests."""

import pytest

from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_estimator_uses_profiling_when_hit():
    """Estimator should use profiled runtime when key is present."""
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.5})
    request = RequestMeta(request_id="r1", width=1280, height=720, num_inference_steps=50)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.0)

    assert estimate == pytest.approx(2.5)


def test_runtime_estimator_falls_back_to_ewma_when_miss():
    """Estimator should fallback to EWMA when profiling key misses."""
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.5})
    request = RequestMeta(request_id="r2", width=1920, height=1080, num_inference_steps=30)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.3)

    assert estimate == pytest.approx(1.3)


def test_runtime_estimator_falls_back_when_no_profiling():
    """Estimator should fallback to EWMA when profiling is unavailable."""
    estimator = RuntimeEstimator()
    request = RequestMeta(request_id="r3", width=1920, height=1080, num_inference_steps=30)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.8)

    assert estimate == pytest.approx(0.8)
