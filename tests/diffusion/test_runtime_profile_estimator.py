# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm_omni.diffusion.runtime_profile import RuntimeProfileEstimator

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_runtime_profile_estimator_uses_exact_profile_hit(tmp_path):
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "instance_type": "img-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 10,
                        "latency_ms": 400,
                    },
                    {
                        "instance_type": "img-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 30,
                        "latency_ms": 1200,
                    },
                    {
                        "instance_type": "img-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 50,
                        "latency_ms": 2000,
                    },
                ]
            }
        )
    )

    estimator = RuntimeProfileEstimator.from_path(str(profile_path), instance_type="img-a")

    assert estimator.estimate_runtime_s(
        task_type="image",
        width=1024,
        height=1024,
        num_frames=1,
        steps=30,
        fallback_s=9.0,
    ) == pytest.approx(1.2)


def test_runtime_profile_estimator_interpolates_steps(tmp_path):
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            [
                {"task_type": "video", "width": 1280, "height": 720, "num_frames": 16, "steps": 10, "latency_s": 1.0},
                {"task_type": "video", "width": 1280, "height": 720, "num_frames": 16, "steps": 30, "latency_s": 3.0},
                {"task_type": "video", "width": 1280, "height": 720, "num_frames": 16, "steps": 50, "latency_s": 5.0},
            ]
        )
    )

    estimator = RuntimeProfileEstimator.from_path(str(profile_path))

    assert estimator.estimate_runtime_s(
        task_type="video",
        width=1280,
        height=720,
        num_frames=16,
        steps=20,
        fallback_s=8.0,
    ) == pytest.approx(2.0)


def test_runtime_profile_estimator_falls_back_to_scaled_nearest_profile(tmp_path):
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            [
                {"task_type": "image", "width": 1024, "height": 1024, "steps": 10, "latency_s": 1.0},
                {"task_type": "image", "width": 1024, "height": 1024, "steps": 50, "latency_s": 5.0},
            ]
        )
    )

    estimator = RuntimeProfileEstimator.from_path(str(profile_path))

    estimate = estimator.estimate_runtime_s(
        task_type="image",
        width=512,
        height=512,
        num_frames=1,
        steps=10,
        fallback_s=9.0,
    )

    assert estimate == pytest.approx(0.25)
