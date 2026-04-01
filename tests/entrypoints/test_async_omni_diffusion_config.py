# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand, _create_default_diffusion_stage_cfg

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_stage_config_includes_cache_backend():
    """Ensure cache knobs survive the default diffusion-stage builder."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "cache_backend": "cache_dit",
            "cache_config": '{"Fn_compute_blocks": 2}',
            "vae_use_slicing": True,
            "ulysses_degree": 2,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert stage_cfg["stage_type"] == "diffusion"
    assert engine_args["cache_backend"] == "cache_dit"
    assert engine_args["cache_config"]["Fn_compute_blocks"] == 2
    assert engine_args["vae_use_slicing"] is True
    assert engine_args["parallel_config"].ulysses_degree == 2
    assert engine_args["model_stage"] == "diffusion"


def test_default_cache_config_used_when_missing():
    """Ensure default cache_config is synthesized when only backend is given."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "cache_backend": "cache_dit",
        }
    )[0]

    cache_config = stage_cfg["engine_args"]["cache_config"]
    assert cache_config is not None
    assert cache_config["Fn_compute_blocks"] == 1


def test_default_stage_devices_from_sequence_parallel():
    """Ensure runtime devices reflect computed diffusion world size."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 2,
            "ring_degree": 2,
        }
    )[0]

    assert stage_cfg["runtime"]["devices"] == "0,1,2,3"


def test_default_stage_config_propagates_ulysses_mode():
    """Ensure UAA mode survives default diffusion-stage creation."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 4,
            "ulysses_mode": "advanced_uaa",
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.ulysses_degree == 4
    assert parallel_config.ulysses_mode == "advanced_uaa"


def test_serve_cli_accepts_ulysses_mode():
    """Ensure diffusion serve CLI exposes ulysses_mode and wires it to parallel_config."""
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--usp",
            "4",
            "--ulysses-mode",
            "advanced_uaa",
        ]
    )

    stage_cfg = _create_default_diffusion_stage_cfg(args)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]

    assert args.ulysses_mode == "advanced_uaa"
    assert parallel_config.ulysses_degree == 4
    assert parallel_config.ulysses_mode == "advanced_uaa"


def test_default_stage_config_derives_step_execution_for_step_level_backend():
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "diffusion_scheduler_backend": "step_level_request_scheduler",
            "diffusion_enable_step_chunk": True,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert engine_args["diffusion_scheduler_backend"] == "step_level_request_scheduler"
    assert engine_args["diffusion_enable_step_chunk"] is True
    assert engine_args["step_execution"] is True


def test_step_level_scheduler_requires_step_chunk():
    with pytest.raises(ValueError, match="diffusion_enable_step_chunk"):
        OmniDiffusionConfig(
            diffusion_scheduler_backend="step_level_request_scheduler",
            diffusion_enable_step_chunk=False,
        )


def test_request_scheduler_rejects_step_chunk():
    with pytest.raises(ValueError, match="diffusion_enable_step_chunk=True"):
        OmniDiffusionConfig(
            diffusion_scheduler_backend="request_scheduler",
            diffusion_enable_step_chunk=True,
        )


def test_step_level_scheduler_rejects_non_fcfs_policy():
    with pytest.raises(NotImplementedError, match="fcfs"):
        OmniDiffusionConfig(
            diffusion_scheduler_backend="step_level_request_scheduler",
            diffusion_enable_step_chunk=True,
            instance_scheduler_policy="sjf",
        )
