from __future__ import annotations

import asyncio
import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "diffusion"
MODULE_PATH = BENCHMARK_DIR / "diffusion_benchmark_serving.py"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _load_benchmark_module():
    module_name = "test_diffusion_benchmark_serving_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCHMARK_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


async def _collect_intervals(module, *, request_rate: float, arrival_seed: int | None = None) -> tuple[list[float], list[str]]:
    intervals: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        intervals.append(delay)

    original_sleep = module.asyncio.sleep
    module.asyncio.sleep = _fake_sleep
    try:
        yielded: list[str] = []
        kwargs = {"request_rate": request_rate}
        if arrival_seed is not None:
            kwargs["arrival_seed"] = arrival_seed
        async for req in module.iter_requests(["a", "b", "c", "d"], **kwargs):
            yielded.append(req)
    finally:
        module.asyncio.sleep = original_sleep

    return intervals, yielded


def test_iter_requests_uses_fixed_arrival_seed_by_default() -> None:
    module = _load_benchmark_module()

    random.seed(1)
    default_intervals_a, yielded_a = asyncio.run(_collect_intervals(module, request_rate=0.125))
    random.seed(999)
    default_intervals_b, yielded_b = asyncio.run(_collect_intervals(module, request_rate=0.125))
    explicit_intervals, yielded_explicit = asyncio.run(
        _collect_intervals(
            module,
            request_rate=0.125,
            arrival_seed=module.DEFAULT_ARRIVAL_SEED,
        )
    )

    assert yielded_a == ["a", "b", "c", "d"]
    assert yielded_b == yielded_a == yielded_explicit
    assert default_intervals_a == default_intervals_b == explicit_intervals


def test_iter_requests_allows_overriding_arrival_seed() -> None:
    module = _load_benchmark_module()

    default_intervals, _ = asyncio.run(_collect_intervals(module, request_rate=0.125))
    custom_intervals, _ = asyncio.run(_collect_intervals(module, request_rate=0.125, arrival_seed=7))

    assert default_intervals != custom_intervals
