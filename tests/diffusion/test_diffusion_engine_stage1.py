# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.runtime_profile import RuntimeProfileEstimator, RuntimeProfileRecord

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _make_request():
    return SimpleNamespace(
        prompts=["prompt"],
        request_ids=["req-1"],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            resolution=1024,
        ),
    )


def test_step_merges_scheduler_metrics(monkeypatch):
    engine = object.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = SimpleNamespace(model_class_name="FakeImagePipeline")
    engine.add_req_and_wait_for_response = lambda req: DiffusionOutput(
        output=torch.tensor([1]),
        metrics={
            "scheduler_policy": "fcfs",
            "queue_wait_ms": 3.0,
            "scheduler_execute_ms": 7.0,
            "scheduler_latency_ms": 10.0,
        },
    )

    monkeypatch.setattr(diffusion_engine_module, "supports_audio_output", lambda model_class_name: False)

    outputs = engine.step(_make_request())

    assert len(outputs) == 1
    assert outputs[0].metrics["scheduler_policy"] == "fcfs"
    assert outputs[0].metrics["queue_wait_ms"] == 3.0
    assert outputs[0].metrics["scheduler_latency_ms"] == 10.0
    assert outputs[0].metrics["image_num"] == 1


def test_step_raises_error_with_error_code_and_request_id():
    engine = object.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = SimpleNamespace(model_class_name="FakeImagePipeline")
    engine.add_req_and_wait_for_response = lambda req: DiffusionOutput(
        error="boom",
        error_code="WORKER_EXEC_FAILED",
        request_id="req-1",
    )

    with pytest.raises(RuntimeError, match=r"\[WORKER_EXEC_FAILED\] request_id=req-1 boom"):
        engine.step(_make_request())


def test_engine_estimate_scheduler_facade():
    engine = object.__new__(DiffusionEngine)
    engine.executor = SimpleNamespace(
        scheduler=SimpleNamespace(
            estimate_waiting_queue_len=lambda: 3,
            estimate_scheduler_load=lambda: {
                "waiting_queue_len": 3,
                "active_request_count": 1,
                "paused_context_count": 0,
            },
        )
    )

    assert engine.estimate_waiting_queue_len() == 3
    assert engine.estimate_scheduler_load() == {
        "waiting_queue_len": 3,
        "active_request_count": 1,
        "paused_context_count": 0,
    }


def test_engine_abort_delegates_to_scheduler():
    aborted: list[str] = []
    engine = object.__new__(DiffusionEngine)
    engine.executor = SimpleNamespace(
        scheduler=SimpleNamespace(abort_request=lambda request_id: aborted.append(request_id) or True)
    )

    engine.abort(["req-1", "req-2"])

    assert aborted == ["req-1", "req-2"]


def test_step_returns_unfinished_placeholder_output():
    engine = object.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = SimpleNamespace(model_class_name="FakeImagePipeline")
    engine.add_req_and_wait_for_response = lambda req: DiffusionOutput(
        output=None,
        finished=False,
        request_id="req-1",
        metrics={"executed_steps": 2, "scheduler_policy": "fcfs"},
    )

    outputs = engine.step(_make_request())

    assert len(outputs) == 1
    assert outputs[0].metrics["unfinished"] is True
    assert outputs[0].metrics["executed_steps"] == 2


def test_step_loops_until_finished_when_step_chunk_enabled():
    engine = object.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = SimpleNamespace(
        model_class_name="FakeImagePipeline",
        diffusion_enable_step_chunk=True,
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=2,
        diffusion_small_request_latency_threshold_ms=None,
        diffusion_image_chunk_budget_steps=None,
        diffusion_video_chunk_budget_steps=None,
    )
    engine.runtime_estimator = RuntimeProfileEstimator()
    outputs_seen = iter(
        [
            DiffusionOutput(output=None, finished=False, request_id="req-1", metrics={"executed_steps": 2}),
            DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-1", metrics={"executed_steps": 4}),
        ]
    )
    engine.add_req_and_wait_for_response = lambda req: next(outputs_seen)

    request = _make_request()
    request.sampling_params.num_inference_steps = 4
    request.executed_steps = 0
    request.max_steps_this_turn = None

    outputs = engine.step(request)

    assert len(outputs) == 1
    assert outputs[0].metrics["chunk_count"] == 2
    assert outputs[0].metrics["executed_steps"] == 4


def test_plan_chunk_budget_uses_latency_threshold_for_small_request():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=4,
        diffusion_image_chunk_budget_steps=4,
        diffusion_video_chunk_budget_steps=2,
        diffusion_small_request_latency_threshold_ms=15000.0,
    )
    engine.runtime_estimator = RuntimeProfileEstimator(
        [
            RuntimeProfileRecord(
                task_type="image",
                width=512,
                height=512,
                num_frames=1,
                steps=20,
                latency_s=12.0,
            )
        ]
    )
    request = _make_request()
    request.sampling_params.width = 512
    request.sampling_params.height = 512
    request.sampling_params.num_inference_steps = 20
    request.sampling_params.num_frames = 1
    request.executed_steps = 0

    assert engine._plan_chunk_budget(request) == 20


def test_plan_chunk_budget_uses_video_specific_budget_for_large_video_request():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=4,
        diffusion_image_chunk_budget_steps=4,
        diffusion_video_chunk_budget_steps=1,
        diffusion_small_request_latency_threshold_ms=1000.0,
    )
    engine.runtime_estimator = RuntimeProfileEstimator(
        [
            RuntimeProfileRecord(
                task_type="video",
                width=854,
                height=480,
                num_frames=80,
                steps=4,
                latency_s=8.0,
            )
        ]
    )
    request = _make_request()
    request.sampling_params.width = 854
    request.sampling_params.height = 480
    request.sampling_params.num_inference_steps = 4
    request.sampling_params.num_frames = 80
    request.sampling_params.fps = 16
    request.executed_steps = 0

    assert engine._plan_chunk_budget(request) == 1


def test_plan_chunk_budget_uses_image_specific_budget_for_large_image_request():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=4,
        diffusion_image_chunk_budget_steps=3,
        diffusion_video_chunk_budget_steps=1,
        diffusion_small_request_latency_threshold_ms=1000.0,
    )
    engine.runtime_estimator = RuntimeProfileEstimator(
        [
            RuntimeProfileRecord(
                task_type="image",
                width=1024,
                height=1024,
                num_frames=1,
                steps=20,
                latency_s=20.0,
            )
        ]
    )
    request = _make_request()
    request.sampling_params.width = 1024
    request.sampling_params.height = 1024
    request.sampling_params.num_inference_steps = 20
    request.sampling_params.num_frames = 1
    request.executed_steps = 0

    assert engine._plan_chunk_budget(request) == 3


def test_plan_chunk_budget_falls_back_to_global_budget_when_modality_budget_missing():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=4,
        diffusion_image_chunk_budget_steps=None,
        diffusion_video_chunk_budget_steps=None,
        diffusion_small_request_latency_threshold_ms=1000.0,
    )
    engine.runtime_estimator = RuntimeProfileEstimator(
        [
            RuntimeProfileRecord(
                task_type="image",
                width=1024,
                height=1024,
                num_frames=1,
                steps=20,
                latency_s=20.0,
            )
        ]
    )
    request = _make_request()
    request.sampling_params.width = 1024
    request.sampling_params.height = 1024
    request.sampling_params.num_inference_steps = 20
    request.sampling_params.num_frames = 1
    request.executed_steps = 0

    assert engine._plan_chunk_budget(request) == 4
