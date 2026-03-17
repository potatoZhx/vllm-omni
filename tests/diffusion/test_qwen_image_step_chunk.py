# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.context import DiffusionRequestContext
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import QwenImageCFGParallelMixin
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self.num_inference_steps = 0
        self._step_index = None
        self._begin_index = None

    def set_schedule(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.float32)
        self.sigmas = torch.arange(num_inference_steps + 1, 0, -1, dtype=torch.float32)
        self._step_index = None
        self._begin_index = None

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def step(self, model_output, timestep, sample, return_dict=False):
        if self._step_index is None:
            self._step_index = 0 if self._begin_index is None else self._begin_index
        if self._step_index + 1 >= len(self.sigmas):
            raise IndexError(
                f"index {self._step_index + 1} is out of bounds for dimension 0 with size {len(self.sigmas)}"
            )
        self._step_index += 1
        return (sample + model_output + 1, )


class _FakeQwenPipeline(QwenImageCFGParallelMixin):
    _capture_scheduler_state = QwenImagePipeline._capture_scheduler_state
    _restore_scheduler_state = QwenImagePipeline._restore_scheduler_state
    step_generation = QwenImagePipeline.step_generation

    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.transformer = SimpleNamespace(do_true_cfg=False)
        self._current_timestep = None
        self._interrupt = False

    @property
    def interrupt(self):
        return self._interrupt

    def predict_noise(self, *args, **kwargs):
        hidden_states = kwargs["hidden_states"]
        return torch.zeros_like(hidden_states)


def _make_ctx(pipeline: _FakeQwenPipeline, request_id: str, total_steps: int) -> DiffusionRequestContext:
    pipeline.scheduler.set_schedule(total_steps)
    ctx = DiffusionRequestContext(request_id=request_id, current_step=0, num_inference_steps=total_steps)
    ctx.scheduler_state = pipeline._capture_scheduler_state()
    ctx.extra_model_state = {
        "prompt_embeds": torch.zeros(1, 1, 1),
        "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.int64),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "latents": torch.zeros(1, 1, 1),
        "img_shapes": [[(1, 1, 1)]],
        "txt_seq_lens": [1],
        "negative_txt_seq_lens": None,
        "timesteps": pipeline.scheduler.timesteps.detach().clone(),
        "do_true_cfg": False,
        "guidance": None,
        "true_cfg_scale": 1.0,
        "additional_transformer_kwargs": {},
    }
    return ctx


def test_qwen_step_chunk_restores_scheduler_state_across_interleaved_requests():
    pipeline = _FakeQwenPipeline()
    ctx_a = _make_ctx(pipeline, "req-a", total_steps=7)

    ctx_a, finished = pipeline.step_generation(ctx_a, steps=4)
    assert finished is False
    assert ctx_a.current_step == 4

    ctx_b = _make_ctx(pipeline, "req-b", total_steps=6)
    ctx_b, finished = pipeline.step_generation(ctx_b, steps=4)
    assert finished is False
    assert ctx_b.current_step == 4
    assert pipeline.scheduler.sigmas.numel() == 7

    ctx_a, finished = pipeline.step_generation(ctx_a, steps=3)

    assert finished is True
    assert ctx_a.current_step == 7
    assert pipeline.scheduler.sigmas.numel() == 8
