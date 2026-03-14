# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

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
