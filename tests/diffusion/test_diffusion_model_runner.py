# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.context import DiffusionRequestContext
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    def __init__(self, output):
        self._output = output
        self.forward_calls = 0

    def forward(self, req):
        del req
        self.forward_calls += 1
        return self._output


def _make_request(skip_cache_refresh: bool = True):
    sampling_params = SimpleNamespace(
        generator=None,
        seed=None,
        generator_device=None,
        num_inference_steps=4,
    )
    return SimpleNamespace(
        prompts=["a prompt"],
        sampling_params=sampling_params,
        skip_cache_refresh=skip_cache_refresh,
    )


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=SimpleNamespace(output="ok"))
    runner.cache_backend = cache_backend
    runner.offload_backend = None
    runner.active_contexts = {}
    runner.od_config = SimpleNamespace(
        cache_backend=cache_backend_name,
        enable_cache_dit_summary=enable_cache_dit_summary,
        parallel_config=SimpleNamespace(use_hsdp=False),
        diffusion_enable_step_chunk=False,
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_kv_cache=lambda req, target_device=None: None,
        receive_multi_kv_cache=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def test_execute_model_skips_cache_summary_without_active_cache_backend(monkeypatch):
    """Guard cache diagnostics with runtime backend state to avoid stale-config crashes."""
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == []


def test_execute_model_emits_cache_summary_with_active_cache_dit_backend(monkeypatch):
    class _EnabledCacheBackend:
        def is_enabled(self):
            return True

    runner = _make_runner(cache_backend=_EnabledCacheBackend(), cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == [(runner.pipeline, True)]


def test_load_model_clears_cache_backend_for_unsupported_pipeline(monkeypatch):
    class _DummyLoader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return SimpleNamespace(transformer=torch.nn.Identity())

    class _DummyMemoryProfiler:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummyCacheBackend:
        def __init__(self):
            self.enabled = False

        def enable(self, pipeline):
            del pipeline
            self.enabled = True

    dummy_cache_backend = _DummyCacheBackend()

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend="cache_dit",
        cache_config={},
        model_class_name="NextStep11Pipeline",
        enforce_eager=True,
    )

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _DummyLoader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _DummyMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda od_config, device: None)
    monkeypatch.setattr(
        model_runner_module, "get_cache_backend", lambda cache_backend, cache_config: dummy_cache_backend
    )

    DiffusionModelRunner.load_model(runner)

    assert runner.cache_backend is None
    assert runner.od_config.cache_backend is None
    assert dummy_cache_backend.enabled is False


def _make_diffusion_request(request_id: str = "req-1", num_inference_steps: int = 4) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["prompt"],
        sampling_params=OmniDiffusionSamplingParams(
            resolution=1024,
            num_outputs_per_prompt=1,
            num_inference_steps=num_inference_steps,
        ),
        request_ids=[request_id],
    )


def test_prepare_generation_tracks_context_with_request_metadata():
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    req = _make_diffusion_request()
    req.executed_steps = 2

    ctx = DiffusionModelRunner.prepare_generation(runner, req)

    assert ctx.request_id == "req-1"
    assert ctx.current_step == 2
    assert ctx.num_inference_steps == 4
    assert runner.active_contexts["req-1"] is ctx


def test_step_generation_updates_context_and_clears_finished_request():
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    ctx = DiffusionRequestContext(request_id="req-1", current_step=1, num_inference_steps=3)
    runner.active_contexts["req-1"] = ctx

    next_ctx, finished = DiffusionModelRunner.step_generation(runner, ctx, steps=2)

    assert finished is True
    assert next_ctx.finished is True
    assert next_ctx.current_step == 3
    assert "req-1" not in runner.active_contexts


def test_finalize_generation_uses_pipeline_hook_and_cleans_context():
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    ctx = DiffusionRequestContext(request_id="req-1", current_step=4, num_inference_steps=4, finished=True)
    runner.active_contexts["req-1"] = ctx

    class _FinalizePipeline(_DummyPipeline):
        def finalize_generation(self, finalize_ctx):
            return SimpleNamespace(output="done", request_id=finalize_ctx.request_id)

    runner.pipeline = _FinalizePipeline(output=SimpleNamespace(output="ignored"))

    output = DiffusionModelRunner.finalize_generation(runner, ctx)

    assert output.output == "done"
    assert output.request_id == "req-1"
    assert "req-1" not in runner.active_contexts


def test_abort_generation_cleans_context_and_calls_pipeline_hook():
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.active_contexts["req-1"] = DiffusionRequestContext(request_id="req-1", current_step=0, num_inference_steps=4)
    aborted = []

    class _AbortPipeline(_DummyPipeline):
        def abort_generation(self, request_id):
            aborted.append(request_id)

    runner.pipeline = _AbortPipeline(output=SimpleNamespace(output="ignored"))

    DiffusionModelRunner.abort_generation(runner, "req-1")

    assert aborted == ["req-1"]
    assert "req-1" not in runner.active_contexts


def test_execute_model_returns_unfinished_output_when_step_chunk_enabled(monkeypatch):
    class _StepAwarePipeline(_DummyPipeline):
        def prepare_generation(self, req):
            return DiffusionRequestContext(
                request_id=req.request_ids[0],
                current_step=req.executed_steps,
                num_inference_steps=req.sampling_params.num_inference_steps,
            )

        def step_generation(self, ctx, steps):
            ctx.current_step += steps
            ctx.finished = False
            return ctx, False

        def finalize_generation(self, ctx):
            return SimpleNamespace(output="done", request_id=ctx.request_id, finished=True)

    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.od_config.diffusion_enable_step_chunk = True
    runner.pipeline = _StepAwarePipeline(output=SimpleNamespace(output="ignored"))
    req = _make_diffusion_request(num_inference_steps=4)
    req.max_steps_this_turn = 2

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.finished is False
    assert output.output is None
    assert output.metrics["executed_steps"] == 2
    assert req.executed_steps == 2


def test_execute_model_finalizes_finished_step_chunk(monkeypatch):
    class _StepAwarePipeline(_DummyPipeline):
        def prepare_generation(self, req):
            return DiffusionRequestContext(
                request_id=req.request_ids[0],
                current_step=req.executed_steps,
                num_inference_steps=req.sampling_params.num_inference_steps,
            )

        def step_generation(self, ctx, steps):
            ctx.current_step += steps
            ctx.finished = True
            return ctx, True

        def finalize_generation(self, ctx):
            return SimpleNamespace(output="done", request_id=ctx.request_id, finished=True)

    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.od_config.diffusion_enable_step_chunk = True
    runner.pipeline = _StepAwarePipeline(output=SimpleNamespace(output="ignored"))
    req = _make_diffusion_request(num_inference_steps=4)
    req.max_steps_this_turn = 4

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.finished is True
    assert output.output == "done"
    assert req.executed_steps == 4
