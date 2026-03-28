# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "riverclouds/qwen_image_random"


def _build_stage_cfg(**kwargs):
    omni = object.__new__(AsyncOmni)
    return omni._create_default_diffusion_stage_cfg(kwargs)[0]


def test_default_stage_config_includes_cache_backend():
    """Ensure cache_backend/cache_config are preserved in default diffusion stage."""
    stage_cfg = _build_stage_cfg(
        model=MODEL,
        diffusion_engine_max_concurrency=7,
        cache_backend="cache_dit",
        cache_config='{"Fn_compute_blocks": 2}',
        vae_use_slicing=True,
        ulysses_degree=2,
        instance_scheduler_policy="p95-first",
        instance_scheduler_slo_target_ms=1800.0,
        instance_scheduler_aging_factor=0.25,
        instance_scheduler_p95_first_base_ms=2200.0,
        instance_scheduler_p95_first_min_ms=1200.0,
        instance_scheduler_p95_first_max_ms=6000.0,
        instance_scheduler_p95_first_backlog_alpha=1.5,
        instance_scheduler_p95_first_size_bias=0.1,
        instance_scheduler_p95_first_age_bias=0.2,
        instance_scheduler_p95_first_starvation_threshold_s=3.0,
        instance_scheduler_p95_first_starvation_boost=1.0,
        instance_scheduler_slack_panic_threshold=1.25,
        instance_scheduler_slack_swap_overhead_ms=120.0,
        instance_scheduler_p95_fusion_tail_budget_ratio=0.2,
        instance_scheduler_p95_fusion_heavy_threshold_s=24.0,
        instance_scheduler_p95_fusion_urgent_slack_ratio=0.75,
        instance_scheduler_p95_fusion_promote_wait_s=45.0,
        instance_scheduler_p95_fusion_nonheavy_streak_limit=3,
        instance_scheduler_p95_fusion_growth_every=12,
        instance_scheduler_p95_fusion_borrowed_cap_max=5,
        instance_scheduler_p95_fusion_min_chunk_steps=2,
        instance_scheduler_p95_fusion_max_chunk_steps=6,
        instance_runtime_profile_path="/profile/runtime.json",
        instance_runtime_profile_name="img-a",
        diffusion_enable_step_chunk=True,
        diffusion_enable_chunk_preemption=True,
        diffusion_chunk_budget_steps=6,
        diffusion_image_chunk_budget_steps=5,
        diffusion_video_chunk_budget_steps=2,
        diffusion_small_request_latency_threshold_ms=15000.0,
    )

    engine_args = stage_cfg["engine_args"]

    assert engine_args["cache_backend"] == "cache_dit"
    cache_config = engine_args["cache_config"]
    assert cache_config["Fn_compute_blocks"] == 2
    assert engine_args["vae_use_slicing"] is True
    assert engine_args["instance_scheduler_policy"] == "p95-first"
    assert engine_args["instance_scheduler_slo_target_ms"] == 1800.0
    assert engine_args["instance_scheduler_aging_factor"] == 0.25
    assert engine_args["instance_scheduler_p95_first_base_ms"] == 2200.0
    assert engine_args["instance_scheduler_p95_first_min_ms"] == 1200.0
    assert engine_args["instance_scheduler_p95_first_max_ms"] == 6000.0
    assert engine_args["instance_scheduler_p95_first_backlog_alpha"] == 1.5
    assert engine_args["instance_scheduler_p95_first_size_bias"] == 0.1
    assert engine_args["instance_scheduler_p95_first_age_bias"] == 0.2
    assert engine_args["instance_scheduler_p95_first_starvation_threshold_s"] == 3.0
    assert engine_args["instance_scheduler_p95_first_starvation_boost"] == 1.0
    assert engine_args["instance_scheduler_slack_panic_threshold"] == 1.25
    assert engine_args["instance_scheduler_slack_swap_overhead_ms"] == 120.0
    assert engine_args["instance_scheduler_p95_fusion_tail_budget_ratio"] == 0.2
    assert engine_args["instance_scheduler_p95_fusion_heavy_threshold_s"] == 24.0
    assert engine_args["instance_scheduler_p95_fusion_urgent_slack_ratio"] == 0.75
    assert engine_args["instance_scheduler_p95_fusion_promote_wait_s"] == 45.0
    assert engine_args["instance_scheduler_p95_fusion_nonheavy_streak_limit"] == 3
    assert engine_args["instance_scheduler_p95_fusion_growth_every"] == 12
    assert engine_args["instance_scheduler_p95_fusion_borrowed_cap_max"] == 5
    assert engine_args["instance_scheduler_p95_fusion_min_chunk_steps"] == 2
    assert engine_args["instance_scheduler_p95_fusion_max_chunk_steps"] == 6
    assert engine_args["instance_runtime_profile_path"] == "/profile/runtime.json"
    assert engine_args["instance_runtime_profile_name"] == "img-a"
    assert engine_args["diffusion_engine_max_concurrency"] == 7
    assert engine_args["diffusion_enable_step_chunk"] is True
    assert engine_args["diffusion_enable_chunk_preemption"] is True
    assert engine_args["diffusion_chunk_budget_steps"] == 6
    assert engine_args["diffusion_image_chunk_budget_steps"] == 5
    assert engine_args["diffusion_video_chunk_budget_steps"] == 2
    assert engine_args["diffusion_small_request_latency_threshold_ms"] == 15000.0
    parallel_config = engine_args["parallel_config"]
    ulysses_degree = getattr(parallel_config, "ulysses_degree", None)
    assert ulysses_degree == 2


def test_chunk_preemption_requires_step_chunk():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="diffusion_enable_chunk_preemption requires diffusion_enable_step_chunk=True"):
        OmniDiffusionConfig(diffusion_enable_step_chunk=False, diffusion_enable_chunk_preemption=True)


def test_small_request_latency_threshold_must_be_positive():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="diffusion_small_request_latency_threshold_ms must be > 0"):
        OmniDiffusionConfig(diffusion_small_request_latency_threshold_ms=0.0)


def test_p95_first_enables_step_chunk_and_chunk_preemption_by_default():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(instance_scheduler_policy="p95-first")

    assert config.diffusion_enable_step_chunk is True
    assert config.diffusion_enable_chunk_preemption is True


def test_slack_hybrid_enables_step_chunk_and_chunk_preemption_by_default():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(instance_scheduler_policy="slack_hybrid")

    assert config.diffusion_enable_step_chunk is True
    assert config.diffusion_enable_chunk_preemption is True


def test_p95_fusion_enables_step_chunk_and_chunk_preemption_by_default():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(instance_scheduler_policy="p95-fusion")

    assert config.diffusion_enable_step_chunk is True
    assert config.diffusion_enable_chunk_preemption is True


@pytest.mark.parametrize("policy", ["p95-first", "sjf_aging", "sjf_aging_guarded", "bypass_guard_sjf", "size_bucket_sjf_aging", "slack_age", "slack_cost_age", "slack_hybrid", "p95-fusion"])
def test_new_instance_scheduler_policies_are_accepted(policy: str):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(instance_scheduler_policy=policy)

    assert config.instance_scheduler_policy == policy


def test_default_cache_config_used_when_missing():
    """Ensure default cache_config is applied when cache_backend is set."""
    stage_cfg = _build_stage_cfg(
        model=MODEL,
        diffusion_engine_max_concurrency=7,
        cache_backend="cache_dit",
    )

    engine_args = stage_cfg["engine_args"]
    cache_config = engine_args["cache_config"]
    assert cache_config is not None
    assert cache_config["Fn_compute_blocks"] == 1


def test_default_stage_devices_from_sequence_parallel():
    """Ensure devices list reflects sequence parallel size when no parallel_config is provided."""
    stage_cfg = _build_stage_cfg(
        model=MODEL,
        ulysses_degree=2,
        ring_degree=2,
    )

    runtime = stage_cfg["runtime"]
    devices = runtime["devices"]
    assert devices == "0,1,2,3"
