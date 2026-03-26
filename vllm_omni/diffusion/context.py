# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass
class DiffusionRequestContext:
    request_id: str
    current_step: int
    num_inference_steps: int
    latents: Any | None = None
    timesteps: Any | None = None
    generator_state: Any | None = None
    scheduler_state: Any | None = None
    prompt_embeds: Any | None = None
    pooled_prompt_embeds: Any | None = None
    extra_model_state: dict[str, Any] = field(default_factory=dict)
    finished: bool = False

    @classmethod
    def from_request(cls, req: OmniDiffusionRequest) -> "DiffusionRequestContext":
        return cls(
            request_id=req.primary_request_id,
            current_step=req.executed_steps,
            num_inference_steps=req.sampling_params.num_inference_steps,
        )
