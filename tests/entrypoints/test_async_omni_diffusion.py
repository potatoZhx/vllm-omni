# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm_omni.entrypoints.async_omni_diffusion as async_diffusion_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_async_omni_diffusion_uses_configured_max_concurrency(monkeypatch):
    monkeypatch.setattr(
        async_diffusion_module,
        "get_hf_file_to_dict",
        lambda filename, model: {"_class_name": "FakeImagePipeline"} if filename == "model_index.json" else {},
    )
    monkeypatch.setattr(
        async_diffusion_module.DiffusionEngine,
        "make_engine",
        staticmethod(lambda od_config: SimpleNamespace(close=lambda: None)),
    )

    od_config = OmniDiffusionConfig.from_kwargs(
        model="fake-model",
        diffusion_engine_max_concurrency=7,
    )

    engine = AsyncOmniDiffusion(model="fake-model", od_config=od_config)
    try:
        assert engine._executor._max_workers == 7  # noqa: SLF001
    finally:
        engine.close()
