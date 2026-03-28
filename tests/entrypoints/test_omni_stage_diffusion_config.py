# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.omni_stage import _build_od_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_od_config_includes_diffusion_fields():
    engine_args = {
        "cache_backend": "cache_dit",
        "cache_config": {"Fn_compute_blocks": 2},
        "vae_use_slicing": True,
        "instance_scheduler_policy": "p95-first",
        "instance_scheduler_slo_target_ms": 1800.0,
        "instance_scheduler_p95_first_base_ms": 2200.0,
        "instance_scheduler_p95_first_backlog_alpha": 1.5,
        "instance_runtime_profile_path": "/profile/runtime.json",
        "instance_runtime_profile_name": "img-a",
    }
    od_config = _build_od_config(engine_args, model="dummy-model")

    assert od_config["model"] == "dummy-model"
    assert od_config["cache_backend"] == "cache_dit"
    assert od_config["cache_config"]["Fn_compute_blocks"] == 2
    assert od_config["vae_use_slicing"] is True
    assert od_config["instance_scheduler_policy"] == "p95-first"
    assert od_config["instance_scheduler_slo_target_ms"] == 1800.0
    assert od_config["instance_scheduler_p95_first_base_ms"] == 2200.0
    assert od_config["instance_scheduler_p95_first_backlog_alpha"] == 1.5
    assert od_config["instance_runtime_profile_path"] == "/profile/runtime.json"
    assert od_config["instance_runtime_profile_name"] == "img-a"


def test_build_od_config_respects_explicit_config():
    engine_args = {
        "od_config": {"cache_backend": "tea_cache"},
        "cache_backend": "cache_dit",
    }
    od_config = _build_od_config(engine_args, model="dummy-model")
    assert od_config == {"cache_backend": "tea_cache"}
