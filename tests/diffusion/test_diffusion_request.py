# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_diffusion_request_sets_stage1_defaults():
    request = OmniDiffusionRequest(
        prompts=["hello"],
        sampling_params=OmniDiffusionSamplingParams(
            resolution=1024,
            num_outputs_per_prompt=1,
            num_inference_steps=4,
        ),
        request_ids=["req-1"],
    )

    assert request.arrival_time > 0
    assert request.request_state == "waiting"
