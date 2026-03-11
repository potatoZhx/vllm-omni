"""Runtime estimator profiling-hit and fallback tests."""

import pytest

from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_estimator_uses_profiling_when_hit():
    """Estimator should use profiled runtime when key is present."""
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.5})
    request = RequestMeta(request_id="r1", width=1280, height=720, num_frames=16, num_inference_steps=50)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.0, instance_type="wan-video-tp2")

    assert estimate == pytest.approx(2.5)


def test_runtime_estimator_falls_back_to_ewma_when_miss():
    """Estimator should fallback to EWMA when profiling key misses."""
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.5})
    request = RequestMeta(request_id="r2", width=1920, height=1080, num_frames=16, num_inference_steps=30)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=1.3, instance_type="wan-video-tp2")

    assert estimate == pytest.approx(1.3)


def test_runtime_estimator_falls_back_when_no_profiling():
    """Estimator should fallback to EWMA when profiling is unavailable."""
    estimator = RuntimeEstimator()
    request = RequestMeta(request_id="r3", width=1920, height=1080, num_frames=16, num_inference_steps=30)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.8, instance_type="wan-video-tp2")

    assert estimate == pytest.approx(0.8)


def test_runtime_estimator_interpolates_between_profiled_steps():
    """Estimator should interpolate when request steps fall between two profiled points."""
    estimator = RuntimeEstimator(
        profiling_data={
            ("wan-video-tp2", 1280, 720, 16, 10): 1.0,
            ("wan-video-tp2", 1280, 720, 16, 30): 3.0,
        }
    )
    request = RequestMeta(request_id="r4", width=1280, height=720, num_frames=16, num_inference_steps=20)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.5, instance_type="wan-video-tp2")

    assert estimate == pytest.approx(2.0)


def test_runtime_estimator_extrapolates_from_profiled_steps():
    """Estimator should extrapolate when request steps are outside the profiled range."""
    estimator = RuntimeEstimator(
        profiling_data={
            ("wan-video-tp2", 1280, 720, 16, 10): 1.0,
            ("wan-video-tp2", 1280, 720, 16, 30): 3.0,
        }
    )
    request = RequestMeta(request_id="r5", width=1280, height=720, num_frames=16, num_inference_steps=50)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.5, instance_type="wan-video-tp2")

    assert estimate == pytest.approx(5.0)


def test_runtime_estimator_scales_when_only_one_profiled_step_exists():
    """Estimator should scale from a single profiled step when no second point exists."""
    estimator = RuntimeEstimator(
        profiling_data={
            ("qwen-image-tp1", 1024, 1024, 1, 10): 0.42,
        }
    )
    request = RequestMeta(request_id="r6", width=1024, height=1024, num_frames=1, num_inference_steps=30)

    estimate = estimator.estimate_runtime_s(request=request, ewma_fallback_s=0.5, instance_type="qwen-image-tp1")

    assert estimate == pytest.approx(1.26)
