"""Runtime estimator profiling-hit and fallback tests."""

import pytest

from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_estimator_uses_profiling_when_hit():
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.5})
    request = RequestMeta(request_id="r1", width=1280, height=720, num_frames=16, num_inference_steps=50)
    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.0, instance_type="wan-video-tp2")
    assert estimate == pytest.approx(2.5)


def test_runtime_estimator_prefers_request_estimated_cost():
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.5})
    request = RequestMeta(
        request_id="r-explicit",
        width=1280,
        height=720,
        num_frames=16,
        num_inference_steps=50,
        estimated_cost_s=3.7,
    )
    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.0, instance_type="wan-video-tp2")
    assert estimate == pytest.approx(3.7)


def test_runtime_estimator_falls_back_to_ewma_when_miss():
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.5})
    request = RequestMeta(request_id="r2", width=1920, height=1080, num_frames=16, num_inference_steps=30)
    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.3, instance_type="wan-video-tp2")
    assert estimate == pytest.approx(1.3)


def test_runtime_estimator_interpolates_between_profiled_steps():
    estimator = RuntimeEstimator(
        profiling_data={
            ("wan-video-tp2", 1280, 720, 16, 10): 1.0,
            ("wan-video-tp2", 1280, 720, 16, 30): 3.0,
        }
    )
    request = RequestMeta(request_id="r4", width=1280, height=720, num_frames=16, num_inference_steps=20)
    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.5, instance_type="wan-video-tp2")
    assert estimate == pytest.approx(2.0)
