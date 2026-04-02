"""Runtime state store bookkeeping tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_omni.global_scheduler.state import RuntimeStateStore
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_store(ewma_alpha: float = 0.2) -> RuntimeStateStore:
    return RuntimeStateStore(
        instances=[
            InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001"),
            InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002"),
        ],
        ewma_alpha=ewma_alpha,
        default_ewma_service_time_s=1.0,
    )


def _request(request_id: str) -> RequestMeta:
    return RequestMeta(request_id=request_id, width=1280, height=720, num_frames=16, num_inference_steps=50)


def test_snapshot_returns_copy():
    store = _make_store()
    snapshot = store.snapshot()
    snapshot["worker-0"].inflight = 100
    assert store.snapshot()["worker-0"].inflight == 0


def test_counters_have_lower_bound_protection():
    store = _make_store()
    store.on_request_finish("worker-0", latency_s=0.5, ok=False)
    store.on_request_finish("worker-0", latency_s=0.2, ok=False)
    stats = store.snapshot()["worker-0"]
    assert stats.queue_len == 0
    assert stats.inflight == 0


def test_ewma_updates_on_finish():
    store = _make_store(ewma_alpha=0.5)
    store.on_request_start("worker-0", _request("r1"))
    stats = store.on_request_finish("worker-0", latency_s=3.0, ok=True, request_id="r1")
    assert stats.ewma_service_time_s == pytest.approx(2.0)


def test_concurrent_start_and_finish_updates_are_consistent():
    store = _make_store()
    operations = 200

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda idx: store.on_request_start("worker-1", _request(f"r{idx}")), range(operations)))

    mid = store.snapshot()["worker-1"]
    assert mid.queue_len == 0
    assert mid.inflight == operations
    assert mid.waiting_requests == ()

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(
            executor.map(
                lambda idx: store.on_request_finish("worker-1", latency_s=1.0, ok=True, request_id=f"r{idx}"),
                range(operations),
            )
        )

    final = store.snapshot()["worker-1"]
    assert final.queue_len == 0
    assert final.inflight == 0
    assert final.outstanding_runtime_s == pytest.approx(0.0)


def test_unknown_instance_raises_key_error():
    store = _make_store()
    with pytest.raises(KeyError, match="Unknown instance id"):
        store.on_request_start("missing-worker", _request("missing"))


def test_sync_instances_adds_and_removes_idle_instances():
    store = _make_store()
    store.sync_instances([InstanceSpec(id="worker-2", endpoint="http://127.0.0.1:9003")])
    snapshot = store.snapshot()
    assert "worker-0" not in snapshot
    assert "worker-1" not in snapshot
    assert "worker-2" in snapshot


def test_sync_instances_keeps_draining_instance_until_finish_converges():
    store = _make_store()
    store.on_request_start("worker-1", _request("r1"))
    store.sync_instances([InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")])
    assert store.snapshot()["worker-1"].inflight == 1

    finished = store.on_request_finish("worker-1", latency_s=0.8, ok=False, request_id="r1")
    assert finished.inflight == 0
    assert "worker-1" not in store.snapshot()


def test_start_and_finish_track_outstanding_runtime_by_request_id():
    store = _make_store()
    request = RequestMeta(request_id="r-cost", estimated_cost_s=2.5)
    started = store.on_request_start("worker-0", request)
    finished = store.on_request_finish("worker-0", latency_s=0.4, ok=True, request_id="r-cost")
    assert started.outstanding_runtime_s == pytest.approx(2.5)
    assert finished.outstanding_runtime_s == pytest.approx(0.0)
