"""Runtime state store concurrency and reconciliation tests."""

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from vllm_omni.global_scheduler.state import RuntimeStateStore
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_store(ewma_alpha: float = 0.2) -> RuntimeStateStore:
    """Create a runtime store fixture with two stable worker ids.

    Args:
        ewma_alpha: EWMA smoothing factor used by the test case.

    Returns:
        Runtime state store preloaded with two instances.
    """
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
    """Snapshot should return detached copies, not mutable internal objects."""
    store = _make_store()
    snapshot = store.snapshot()
    snapshot["worker-0"].queue_len = 100

    fresh = store.snapshot()
    assert fresh["worker-0"].queue_len == 0


def test_counters_have_lower_bound_protection():
    """Finish calls should not drive queue/inflight counters below zero."""
    store = _make_store()

    store.on_request_finish("worker-0", latency_s=0.5, ok=False)
    store.on_request_finish("worker-0", latency_s=0.2, ok=False)

    stats = store.snapshot()["worker-0"]
    assert stats.queue_len == 0
    assert stats.inflight == 0


def test_ewma_updates_on_finish():
    """EWMA service time should update on request finish with latency."""
    store = _make_store(ewma_alpha=0.5)
    store.on_request_start("worker-0", _request("r1"))
    stats = store.on_request_finish("worker-0", latency_s=3.0, ok=True)

    assert stats.ewma_service_time_s == pytest.approx(2.0)


def test_concurrent_start_and_finish_updates_are_consistent():
    """Concurrent start/finish updates should remain counter-consistent."""
    store = _make_store()
    operations = 200

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda idx: store.on_request_start("worker-1", _request(f"r{idx}")), range(operations)))

    mid = store.snapshot()["worker-1"]
    assert mid.queue_len == operations - 1
    assert mid.inflight == 1
    assert len(mid.waiting_requests) == operations - 1

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda _: store.on_request_finish("worker-1", latency_s=1.0, ok=True), range(operations)))

    final = store.snapshot()["worker-1"]
    assert final.queue_len == 0
    assert final.inflight == 0
    assert final.waiting_requests == ()


def test_unknown_instance_raises_key_error():
    """Unknown instance ids should raise KeyError on state updates."""
    store = _make_store()

    with pytest.raises(KeyError, match="Unknown instance id"):
        store.on_request_start("missing-worker", _request("missing"))


def test_sync_instances_adds_and_removes_idle_instances():
    """sync_instances should remove idle removed ids and add new ids."""
    store = _make_store()

    store.sync_instances([InstanceSpec(id="worker-2", endpoint="http://127.0.0.1:9003")])
    snapshot = store.snapshot()

    assert "worker-0" not in snapshot
    assert "worker-1" not in snapshot
    assert "worker-2" in snapshot


def test_sync_instances_keeps_draining_instance_until_finish_converges():
    """Draining instances should be retained until in-flight work converges."""
    store = _make_store()
    store.on_request_start("worker-1", _request("r1"))

    store.sync_instances([InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")])
    after_sync = store.snapshot()
    assert "worker-1" in after_sync
    assert after_sync["worker-1"].inflight == 1

    finished = store.on_request_finish("worker-1", latency_s=0.8, ok=False)
    assert finished.queue_len == 0
    assert finished.inflight == 0

    final_snapshot = store.snapshot()
    assert "worker-1" not in final_snapshot


def test_start_over_capacity_tracks_fifo_waiting_queue():
    """Requests beyond max concurrency should remain in FIFO waiting state."""
    store = RuntimeStateStore(
        instances=[
            InstanceSpec(
                id="worker-0",
                endpoint="http://127.0.0.1:9001",
                launch_args=["--diffusion-engine-max-concurrency", "2"],
            )
        ]
    )

    store.on_request_start("worker-0", _request("r1"))
    store.on_request_start("worker-0", _request("r2"))
    stats = store.on_request_start("worker-0", _request("r3"))

    assert stats.inflight == 2
    assert stats.queue_len == 1
    assert [request.request_id for request in stats.waiting_requests] == ["r3"]


def test_finish_promotes_waiting_request_to_inflight():
    """Finishing one running request should promote the FIFO head from waiting."""
    store = RuntimeStateStore(
        instances=[
            InstanceSpec(
                id="worker-0",
                endpoint="http://127.0.0.1:9001",
                launch_args=["--diffusion-engine-max-concurrency", "1"],
            )
        ]
    )

    store.on_request_start("worker-0", _request("r1"))
    store.on_request_start("worker-0", _request("r2"))
    stats = store.on_request_finish("worker-0", latency_s=0.5, ok=True)

    assert stats.inflight == 1
    assert stats.queue_len == 0
    assert stats.waiting_requests == ()


def test_wait_for_available_capacity_unblocks_after_finish():
    """Capacity wait should block on full worker and unblock after finish."""
    store = RuntimeStateStore(
        instances=[
            InstanceSpec(
                id="worker-0",
                endpoint="http://127.0.0.1:9001",
                launch_args=["--diffusion-engine-max-concurrency", "1"],
            )
        ]
    )

    store.on_request_start("worker-0", _request("r1"))
    assert store.wait_for_available_capacity(["worker-0"], timeout_s=0.01) is False

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.wait_for_available_capacity, ["worker-0"], 1.0)
        time.sleep(0.05)
        store.on_request_finish("worker-0", latency_s=0.2, ok=True)
        assert future.result(timeout=1.0) is True
