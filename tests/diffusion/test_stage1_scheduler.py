# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections import deque
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.stage1_scheduler import Stage1Scheduler, _QueuedRequest

pytestmark = [pytest.mark.diffusion]


def _tagged_output(tag: str) -> DiffusionOutput:
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(
    tag: str,
    *,
    num_inference_steps: int = 1,
    resolution: int = 1024,
    num_outputs_per_prompt: int = 1,
    extra_args: dict | None = None,
) -> Mock:
    req = Mock()
    req.request_ids = [tag]
    req.sampling_params = SimpleNamespace(
        num_inference_steps=num_inference_steps,
        resolution=resolution,
        num_outputs_per_prompt=num_outputs_per_prompt,
        num_frames=1,
        width=None,
        height=None,
        extra_args=extra_args or {},
    )
    return req


def _make_stage1_scheduler(
    *,
    policy: str = "fcfs",
    slo_target_ms: float | None = None,
    aging_factor: float = 0.0,
    profile_path: str | None = None,
    profile_name: str | None = None,
    slack_panic_threshold: float = 1.0,
    slack_swap_overhead_ms: float = 0.0,
):
    sched = Stage1Scheduler()
    sched.num_workers = 1
    sched.od_config = SimpleNamespace(
        num_gpus=1,
        instance_scheduler_policy=policy,
        instance_scheduler_slo_target_ms=slo_target_ms,
        instance_scheduler_slo_floor_ms=0.0,
        instance_scheduler_aging_factor=aging_factor,
        instance_scheduler_p95_first_base_ms=None,
        instance_scheduler_p95_first_min_ms=0.0,
        instance_scheduler_p95_first_max_ms=None,
        instance_scheduler_p95_first_backlog_alpha=1.0,
        instance_scheduler_p95_first_size_bias=0.0,
        instance_scheduler_p95_first_age_bias=0.0,
        instance_scheduler_p95_first_starvation_threshold_s=None,
        instance_scheduler_p95_first_starvation_boost=0.0,
        instance_scheduler_p95_bucket_count=4,
        instance_scheduler_p95_bucket_min_window_ms=200.0,
        instance_scheduler_p95_bucket_starvation_threshold_s=None,
        instance_scheduler_p95_bucket_starvation_promote_levels=1,
        instance_scheduler_slack_panic_threshold=slack_panic_threshold,
        instance_scheduler_slack_swap_overhead_ms=slack_swap_overhead_ms,
        instance_runtime_profile_path=profile_path,
        instance_runtime_profile_name=profile_name,
    )
    sched.initialize(sched.od_config)

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_mq = Mock()
    mock_mq.enqueue = req_q.put

    mock_rmq = Mock()
    mock_rmq.dequeue = lambda timeout=None: res_q.get(timeout=timeout if timeout else 10)

    sched.mq = mock_mq
    sched.result_mq = mock_rmq
    return sched, req_q, res_q


def test_stage1_scheduler_attaches_metrics_on_success():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-success")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1])))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.request_id == "req-success"
    assert output.metrics["scheduler_policy"] == "fcfs"
    assert output.metrics["scheduler_latency_ms"] >= 0
    assert output.metrics["queue_wait_ms"] >= 0
    assert output.metrics["scheduler_execute_ms"] >= 0
    assert output.metrics["width"] == 1024
    assert output.metrics["height"] == 1024
    assert output.metrics["total_steps"] == 1
    assert output.metrics["executed_steps"] == 1
    assert output.metrics["remaining_steps"] == 0
    assert req.first_enqueue_time is not None
    assert req.first_dispatch_time is not None
    assert req.completion_time is not None


def test_stage1_scheduler_normalizes_worker_error_dict():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-fail")

    def _worker():
        req_q.get(timeout=5)
        res_q.put({"status": "error", "error": "boom"})

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error == "boom"
    assert output.error_code == "WORKER_EXEC_FAILED"
    assert output.request_id == "req-fail"
    assert output.metrics["scheduler_policy"] == "fcfs"


def test_stage1_scheduler_preserves_fcfs_order():
    sched, req_q, res_q = _make_stage1_scheduler()
    seen_request_ids: list[str] = []

    def _worker():
        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            seen_request_ids.append(rpc_request["args"][0].request_ids[0])
            request_id = rpc_request["args"][0].request_ids[0]
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    outputs: dict[str, DiffusionOutput] = {}

    def _run(tag: str):
        outputs[tag] = sched.add_req(_mock_request(tag))

    t1 = threading.Thread(target=_run, args=("req-1",), daemon=True)
    t2 = threading.Thread(target=_run, args=("req-2",), daemon=True)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)
    worker.join(5)

    assert seen_request_ids == ["req-1", "req-2"]
    assert outputs["req-1"].request_id == "req-1"
    assert outputs["req-2"].request_id == "req-2"


def test_stage1_scheduler_keeps_request_running_for_unfinished_output():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-chunk", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=None, finished=False, metrics={"executed_steps": 2}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.finished is False
    assert req.request_state == "waiting"
    assert req.executed_steps == 2
    assert output.metrics["executed_steps"] == 2
    assert output.metrics["remaining_steps"] == 2
    assert req.last_preempted_time is not None
    with sched._queue_cv:
        assert len(sched._waiting_queue) == 1
        assert sched._waiting_queue[0].request is req


def test_stage1_scheduler_reuses_waiting_entry_for_resumed_request():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-resume", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=None, finished=False, metrics={"executed_steps": 2}))
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, metrics={"executed_steps": 4}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    first_output = sched.add_req(req)
    assert first_output.finished is False

    enqueue_calls: list[str] = []
    original_enqueue = sched._enqueue_request_locked

    def _record_enqueue(request):
        enqueue_calls.append(request.request_ids[0])
        return original_enqueue(request)

    sched._enqueue_request_locked = _record_enqueue
    second_output = sched.add_req(req)
    worker.join(5)

    assert second_output.finished is True
    assert enqueue_calls == []


def test_stage1_scheduler_sjf_prioritizes_requeued_shorter_request_over_new_longer_request():
    sched, req_q, res_q = _make_stage1_scheduler(policy="sjf")
    req1 = _mock_request("req-1", num_inference_steps=41, extra_args={"estimated_cost_s": 2.0})
    req2 = _mock_request("req-2", num_inference_steps=28, extra_args={"estimated_cost_s": 10.0})
    dispatch_order: list[str] = []
    req1_requeued = threading.Event()
    allow_req1_resume = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        dispatch_order.append(first["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=None, finished=False, request_id="req-1", metrics={"executed_steps": 30}))

        second = req_q.get(timeout=5)
        dispatch_order.append(second["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-1", metrics={"executed_steps": 41}))

        third = req_q.get(timeout=5)
        dispatch_order.append(third["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-2", metrics={"executed_steps": 28}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    def _run_req1():
        first_output = sched.add_req(req1)
        results["req-1-first"] = first_output
        req1_requeued.set()
        allow_req1_resume.wait(timeout=5)
        results["req-1-second"] = sched.add_req(req1)

    req1_thread = threading.Thread(target=_run_req1, daemon=True)
    req1_thread.start()
    assert req1_requeued.wait(timeout=5)

    req2_thread = threading.Thread(
        target=lambda: results.setdefault("req-2", sched.add_req(req2)),
        daemon=True,
    )
    req2_thread.start()

    time.sleep(0.1)
    allow_req1_resume.set()

    req1_thread.join(5)
    req2_thread.join(5)
    worker.join(5)

    assert dispatch_order == ["req-1", "req-1", "req-2"]
    assert results["req-1-first"].finished is False
    assert results["req-1-second"].finished is True
    assert results["req-2"].finished is True


def test_stage1_scheduler_marks_finished_request_as_fully_executed_without_worker_metric():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-finished", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.finished is True
    assert req.request_state == "finished"
    assert req.executed_steps == 4
    assert output.metrics["executed_steps"] == 4
    assert output.metrics["remaining_steps"] == 0


def test_stage1_scheduler_slo_first_reorders_waiting_queue_by_slack_over_remaining_cost():
    sched, req_q, res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    active = threading.Thread(
        target=lambda: results.setdefault(
            "active",
            sched.add_req(_mock_request("active", num_inference_steps=1, extra_args={"slo_ms": 5000.0})),
        ),
        daemon=True,
    )
    tighter = threading.Thread(
        target=lambda: results.setdefault(
            "tighter",
            sched.add_req(
                _mock_request(
                    "tighter",
                    num_inference_steps=2,
                    extra_args={"slo_ms": 3000.0, "estimated_cost_s": 2.0},
                )
            ),
        ),
        daemon=True,
    )
    looser = threading.Thread(
        target=lambda: results.setdefault(
            "looser",
            sched.add_req(
                _mock_request(
                    "looser",
                    num_inference_steps=1,
                    extra_args={"slo_ms": 2500.0, "estimated_cost_s": 1.0},
                )
            ),
        ),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    tighter.start()
    looser.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    tighter.join(5)
    looser.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "looser", "tighter"]
    assert results["looser"].metrics["scheduler_policy"] == "slo_first"
    assert results["looser"].metrics["self_hit"] == 1
    assert results["looser"].metrics["queue_reorder_count"] == 1


def test_stage1_scheduler_slo_first_splits_on_time_and_best_effort_sets():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-4", num_inference_steps=4, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-3", num_inference_steps=3, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-2", num_inference_steps=2, extra_args={"slo_ms": 5000.0})
        )
        last = sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-1", num_inference_steps=1, extra_args={"slo_ms": 5000.0})
        )

        ordered = [queued.request.request_ids[0] for queued in sched._waiting_queue]  # noqa: SLF001

    assert ordered == ["cost-2", "cost-1", "cost-3", "cost-4"]
    assert last.schedule_metrics["on_time_set_size"] == 2
    assert last.schedule_metrics["best_effort_set_size"] == 2
    assert last.schedule_metrics["dispatch_group"] == "on_time"


def test_stage1_scheduler_slo_first_orders_on_time_set_by_slack_over_remaining_cost():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    tighter = _mock_request(
        "tighter",
        num_inference_steps=2,
        extra_args={"slo_ms": 3000.0, "estimated_cost_s": 2.0},
    )
    looser = _mock_request(
        "looser",
        num_inference_steps=1,
        extra_args={"slo_ms": 2500.0, "estimated_cost_s": 1.0},
    )

    with sched._queue_cv:  # noqa: SLF001
        q1 = sched._enqueue_request_locked(tighter)  # noqa: SLF001
        q2 = sched._enqueue_request_locked(looser)  # noqa: SLF001
        plan = sched._build_waiting_plan(list(sched._waiting_queue), now=q1.enqueue_time)  # noqa: SLF001

    assert q1.sequence_id in plan.feasible_ids
    assert q2.sequence_id in plan.feasible_ids
    assert [queued.request.request_ids[0] for queued in plan.on_time_queue] == [
        "tighter",
        "looser",
    ]


def test_stage1_scheduler_slack_age_prefers_older_request_when_slack_is_tied():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slack_age", slo_target_ms=5000.0, aging_factor=1.0)
    older = _mock_request("older", num_inference_steps=2, extra_args={"deadline_ts": 24.0, "estimated_cost_s": 2.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"deadline_ts": 23.0, "estimated_cost_s": 1.0})
    older.arrival_time = 10.0
    newer.arrival_time = 15.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.ordered_queue] == ["older", "newer"]
    assert [queued.request.request_ids[0] for queued in plan.on_time_queue] == ["older", "newer"]
    assert plan.best_effort_queue == []


@pytest.mark.parametrize("policy", ["slack_age", "slack_cost_age"])
def test_stage1_scheduler_slack_policies_use_single_queue_ranking(policy: str):
    sched, _req_q, _res_q = _make_stage1_scheduler(policy=policy, slo_target_ms=5000.0, aging_factor=1.0)
    late_old = _mock_request("late-old", num_inference_steps=1, extra_args={"deadline_ts": 18.0, "estimated_cost_s": 1.0})
    on_time_new = _mock_request(
        "on-time-new",
        num_inference_steps=1,
        extra_args={"deadline_ts": 30.0, "estimated_cost_s": 1.0},
    )
    late_old.arrival_time = 10.0
    on_time_new.arrival_time = 20.0

    with sched._queue_cv:  # noqa: SLF001
        first = sched._enqueue_request_locked(late_old)  # noqa: SLF001
        second = sched._enqueue_request_locked(on_time_new)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.ordered_queue] == ["late-old", "on-time-new"]
    assert plan.best_effort_queue == []
    assert plan.feasible_ids == {first.sequence_id, second.sequence_id}
    assert second.schedule_metrics["dispatch_group"] == "single_queue"


def test_stage1_scheduler_slack_cost_age_penalizes_large_request_when_slack_and_age_match():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slack_cost_age", slo_target_ms=5000.0, aging_factor=0.0)
    large = _mock_request("large", num_inference_steps=4, extra_args={"slo_ms": 8000.0, "estimated_cost_s": 4.0})
    small = _mock_request("small", num_inference_steps=1, extra_args={"slo_ms": 5000.0, "estimated_cost_s": 1.0})
    large.arrival_time = 10.0
    small.arrival_time = 10.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(large)  # noqa: SLF001
        sched._enqueue_request_locked(small)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=10.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.ordered_queue] == ["small", "large"]
    assert plan.best_effort_queue == []


def test_stage1_scheduler_slack_hybrid_uses_throughput_srpt_with_aging_in_safe_mode():
    sched, _req_q, _res_q = _make_stage1_scheduler(
        policy="slack_hybrid",
        aging_factor=1.0,
        slack_panic_threshold=1.0,
    )
    now = time.monotonic()
    older = _mock_request("older", num_inference_steps=10, extra_args={"deadline_ts": now + 40.0, "estimated_cost_s": 10.0})
    newer = _mock_request("newer", num_inference_steps=3, extra_args={"deadline_ts": now + 5.0, "estimated_cost_s": 3.0})
    older.arrival_time = now - 20.0
    newer.arrival_time = now

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        ordered = list(sched._waiting_queue)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in ordered] == ["older", "newer"]
    assert ordered[0].schedule_metrics["hybrid_mode"] == "throughput_srpt"
    assert ordered[0].schedule_metrics["throughput_priority"] < ordered[1].schedule_metrics["throughput_priority"]


def test_stage1_scheduler_slack_hybrid_switches_to_panic_edf_for_urgent_request():
    sched, _req_q, _res_q = _make_stage1_scheduler(
        policy="slack_hybrid",
        aging_factor=0.0,
        slack_panic_threshold=1.0,
    )
    now = time.monotonic()
    urgent = _mock_request("urgent", num_inference_steps=2, extra_args={"deadline_ts": now + 1.5, "estimated_cost_s": 2.0})
    short_safe = _mock_request("short-safe", num_inference_steps=1, extra_args={"deadline_ts": now + 10.0, "estimated_cost_s": 1.0})
    urgent.arrival_time = now
    short_safe.arrival_time = now

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(short_safe)  # noqa: SLF001
        sched._enqueue_request_locked(urgent)  # noqa: SLF001
        ordered = list(sched._waiting_queue)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in ordered] == ["urgent", "short-safe"]
    assert ordered[0].schedule_metrics["hybrid_mode"] == "panic_edf"
    assert ordered[0].schedule_metrics["is_urgent"] == 1


def test_stage1_scheduler_slack_hybrid_adapts_to_step_chunk_requeue():
    sched, req_q, res_q = _make_stage1_scheduler(
        policy="slack_hybrid",
        aging_factor=0.0,
        slack_panic_threshold=1.0,
    )
    now = time.monotonic()
    req1 = _mock_request("req-1", num_inference_steps=20, extra_args={"deadline_ts": now + 100.0, "estimated_cost_s": 20.0})
    req2 = _mock_request("req-2", num_inference_steps=2, extra_args={"deadline_ts": now + 2.0, "estimated_cost_s": 2.0})
    req1.arrival_time = now - 10.0
    req2.arrival_time = now
    dispatch_order: list[str] = []
    req1_requeued = threading.Event()
    allow_req1_resume = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        dispatch_order.append(first["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=None, finished=False, request_id="req-1", metrics={"executed_steps": 10}))

        second = req_q.get(timeout=5)
        dispatch_order.append(second["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-2", metrics={"executed_steps": 2}))

        third = req_q.get(timeout=5)
        dispatch_order.append(third["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-1", metrics={"executed_steps": 20}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    def _run_req1():
        first_output = sched.add_req(req1)
        results["req-1-first"] = first_output
        req1_requeued.set()
        allow_req1_resume.wait(timeout=5)
        results["req-1-second"] = sched.add_req(req1)

    req1_thread = threading.Thread(target=_run_req1, daemon=True)
    req1_thread.start()
    assert req1_requeued.wait(timeout=5)

    req2_thread = threading.Thread(
        target=lambda: results.setdefault("req-2", sched.add_req(req2)),
        daemon=True,
    )
    req2_thread.start()

    time.sleep(0.1)
    allow_req1_resume.set()

    req1_thread.join(5)
    req2_thread.join(5)
    worker.join(5)

    assert dispatch_order == ["req-1", "req-2", "req-1"]
    assert results["req-1-first"].finished is False
    assert results["req-2"].metrics["scheduler_policy"] == "slack_hybrid"
    assert results["req-2"].metrics["hybrid_mode"] == "panic_edf"


def test_stage1_scheduler_sjf_uses_remaining_steps():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="sjf")
    long_req = _mock_request("long", num_inference_steps=10)
    short_remaining_req = _mock_request("short-remaining", num_inference_steps=10)
    short_remaining_req.executed_steps = 8

    with sched._queue_cv:
        sched._enqueue_request_locked(long_req)
        sched._enqueue_request_locked(short_remaining_req)

        ordered = [queued.request.request_ids[0] for queued in sched._waiting_queue]

    assert ordered == ["short-remaining", "long"]


def test_stage1_scheduler_sjf_aging_promotes_old_request_over_short_new_request():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="sjf_aging")
    now = time.monotonic()
    older = _mock_request("older", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    older.arrival_time = now - 20.0
    newer.arrival_time = now

    with sched._queue_cv:
        sched._enqueue_request_locked(older)
        sched._enqueue_request_locked(newer)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["older", "newer"]
    assert ordered[0].schedule_metrics["scheduler_policy"] == "sjf_aging"
    assert ordered[0].schedule_metrics["aged_cost_s"] < ordered[1].schedule_metrics["aged_cost_s"]
    assert ordered[0].schedule_metrics["aging_factor"] == pytest.approx(1.0)
    assert ordered[0].schedule_metrics["aging_cost_weight"] == pytest.approx(1.0)


def test_stage1_scheduler_sjf_aging_uses_cost_aware_weight_for_large_request():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="sjf_aging")
    now = time.monotonic()
    older_large = _mock_request("older-large", num_inference_steps=35, extra_args={"estimated_cost_s": 37.0})
    newer_medium = _mock_request("newer-medium", num_inference_steps=25, extra_args={"estimated_cost_s": 12.0})
    older_large.arrival_time = now - 2.0
    newer_medium.arrival_time = now

    with sched._queue_cv:
        sched._enqueue_request_locked(older_large)
        sched._enqueue_request_locked(newer_medium)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["older-large", "newer-medium"]
    assert ordered[0].schedule_metrics["aging_cost_weight"] > ordered[1].schedule_metrics["aging_cost_weight"]
    assert ordered[0].schedule_metrics["aged_cost_s"] < ordered[1].schedule_metrics["aged_cost_s"]


def test_stage1_scheduler_sjf_aging_guarded_promotes_protected_old_large_request():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="sjf_aging_guarded")
    now = time.monotonic()
    older_large = _mock_request("older-large", num_inference_steps=35, extra_args={"estimated_cost_s": 37.0})
    newer_medium = _mock_request("newer-medium", num_inference_steps=25, extra_args={"estimated_cost_s": 12.0})
    older_large.arrival_time = now - 80.0
    newer_medium.arrival_time = now

    with sched._queue_cv:
        sched._enqueue_request_locked(older_large)
        sched._enqueue_request_locked(newer_medium)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["older-large", "newer-medium"]
    assert ordered[0].schedule_metrics["scheduler_policy"] == "sjf_aging_guarded"
    assert ordered[0].schedule_metrics["tail_protected"] == 1
    assert ordered[0].schedule_metrics["dispatch_group"] == "protected"
    assert getattr(ordered[0].request, "tail_protected", False) is True


def test_stage1_scheduler_sjf_aging_adapts_to_step_chunk_requeue():
    sched, req_q, res_q = _make_stage1_scheduler(policy="sjf_aging")
    req1 = _mock_request("req-1", num_inference_steps=20, extra_args={"estimated_cost_s": 20.0})
    req2 = _mock_request("req-2", num_inference_steps=3, extra_args={"estimated_cost_s": 3.0})
    req1.arrival_time = time.monotonic() - 30.0
    dispatch_order: list[str] = []
    req1_requeued = threading.Event()
    allow_req1_resume = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        dispatch_order.append(first["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=None, finished=False, request_id="req-1", metrics={"executed_steps": 10}))

        second = req_q.get(timeout=5)
        dispatch_order.append(second["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-1", metrics={"executed_steps": 20}))

        third = req_q.get(timeout=5)
        dispatch_order.append(third["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-2", metrics={"executed_steps": 3}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    def _run_req1():
        first_output = sched.add_req(req1)
        results["req-1-first"] = first_output
        req1_requeued.set()
        allow_req1_resume.wait(timeout=5)
        results["req-1-second"] = sched.add_req(req1)

    req1_thread = threading.Thread(target=_run_req1, daemon=True)
    req1_thread.start()
    assert req1_requeued.wait(timeout=5)

    req2_thread = threading.Thread(
        target=lambda: results.setdefault("req-2", sched.add_req(req2)),
        daemon=True,
    )
    req2_thread.start()

    time.sleep(0.1)
    allow_req1_resume.set()

    req1_thread.join(5)
    req2_thread.join(5)
    worker.join(5)

    assert dispatch_order == ["req-1", "req-1", "req-2"]
    assert results["req-1-first"].finished is False
    assert results["req-1-second"].finished is True
    assert results["req-1-second"].metrics["scheduler_policy"] == "sjf_aging"
    assert results["req-1-second"].metrics["aged_cost_s"] < results["req-2"].metrics["aged_cost_s"]


def test_stage1_scheduler_size_bucket_sjf_aging_prefers_smaller_bucket_before_larger_bucket():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="size_bucket_sjf_aging")
    smaller = _mock_request("smaller", resolution=768, num_inference_steps=20, extra_args={"estimated_cost_s": 3.0})
    larger = _mock_request("larger", resolution=1536, num_inference_steps=10, extra_args={"estimated_cost_s": 1.0})

    with sched._queue_cv:
        sched._enqueue_request_locked(larger)
        sched._enqueue_request_locked(smaller)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["smaller", "larger"]
    assert ordered[0].schedule_metrics["raw_size_bucket_id"] < ordered[1].schedule_metrics["raw_size_bucket_id"]


def test_stage1_scheduler_size_bucket_sjf_aging_uses_sjf_within_same_bucket():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="size_bucket_sjf_aging")
    longer = _mock_request("longer", resolution=1024, num_inference_steps=25, extra_args={"estimated_cost_s": 4.0})
    shorter = _mock_request("shorter", resolution=1024, num_inference_steps=25, extra_args={"estimated_cost_s": 1.0})

    with sched._queue_cv:
        sched._enqueue_request_locked(longer)
        sched._enqueue_request_locked(shorter)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["shorter", "longer"]
    assert ordered[0].schedule_metrics["raw_size_bucket_id"] == ordered[1].schedule_metrics["raw_size_bucket_id"]
    assert ordered[0].schedule_metrics["aged_cost_s"] < ordered[1].schedule_metrics["aged_cost_s"]


def test_stage1_scheduler_size_bucket_sjf_aging_promotes_old_large_request_across_buckets():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="size_bucket_sjf_aging", aging_factor=1.0)
    now = time.monotonic()
    old_large = _mock_request("old-large", resolution=1536, num_inference_steps=35, extra_args={"estimated_cost_s": 8.0})
    new_medium = _mock_request("new-medium", resolution=768, num_inference_steps=20, extra_args={"estimated_cost_s": 1.0})
    old_large.arrival_time = now - 25.0
    new_medium.arrival_time = now

    with sched._queue_cv:
        sched._enqueue_request_locked(old_large)
        sched._enqueue_request_locked(new_medium)
        ordered = list(sched._waiting_queue)

    assert [queued.request.request_ids[0] for queued in ordered] == ["old-large", "new-medium"]
    assert ordered[0].schedule_metrics["raw_size_bucket_id"] > ordered[0].schedule_metrics["effective_size_bucket_id"]
    assert ordered[0].schedule_metrics["bucket_promotion_levels"] >= 2


def test_stage1_scheduler_deadline_uses_request_arrival_time():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    req = _mock_request("req-deadline", num_inference_steps=10, extra_args={"slo_ms": 2000.0})
    req.arrival_time = 100.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(req)

    assert sched._deadline_ts(queued) == pytest.approx(102.0)


@pytest.mark.parametrize("policy", ["slo_first", "slack_age", "slack_cost_age"])
def test_stage1_scheduler_deadline_aware_policies_fallback_to_learned_p95_without_explicit_deadline(policy: str):
    sched, _req_q, _res_q = _make_stage1_scheduler(policy=policy, slo_target_ms=None)
    for _ in range(20):
        sched._record_p95_first_latency_ms(1500.0)  # noqa: SLF001

    req = _mock_request("req-learned-deadline", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    req.arrival_time = 100.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(req)
        waiting = list(sched._waiting_queue)

    plan = sched._build_waiting_plan(waiting, now=101.0)  # noqa: SLF001

    assert plan.uses_learned_deadline is True
    assert plan.dynamic_p95_ms == pytest.approx(1500.0)
    assert sched._deadline_ts(queued, plan.dynamic_p95_ms) == pytest.approx(101.5)  # noqa: SLF001
    assert queued.schedule_metrics["learned_deadline_fallback"] == 1
    assert queued.schedule_metrics["dynamic_p95_ms"] == pytest.approx(1500.0)


def test_stage1_scheduler_slack_hybrid_fallback_to_learned_p95_without_explicit_deadline():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slack_hybrid", slo_target_ms=None, aging_factor=0.0)
    for _ in range(20):
        sched._record_p95_first_latency_ms(2000.0)  # noqa: SLF001

    older = _mock_request("older", num_inference_steps=3, extra_args={"estimated_cost_s": 3.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    now = time.monotonic()
    older.arrival_time = now - 1.0
    newer.arrival_time = now

    with sched._queue_cv:
        sched._enqueue_request_locked(older)
        sched._enqueue_request_locked(newer)
        ordered = list(sched._waiting_queue)

    assert ordered[0].schedule_metrics["dynamic_p95_ms"] >= 2000.0
    assert ordered[0].schedule_metrics["learned_p95_ms"] == pytest.approx(2000.0)
    assert ordered[0].schedule_metrics["learned_deadline_fallback"] == 1
    assert ordered[0].schedule_metrics["hybrid_mode"] in {"panic_edf", "throughput_srpt"}


def test_stage1_scheduler_sjf_reorders_waiting_queue():
    sched, req_q, res_q = _make_stage1_scheduler(policy="sjf")
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    active = threading.Thread(
        target=lambda: results.setdefault("active", sched.add_req(_mock_request("active", num_inference_steps=1))),
        daemon=True,
    )
    long_waiting = threading.Thread(
        target=lambda: results.setdefault("long", sched.add_req(_mock_request("long", num_inference_steps=20))),
        daemon=True,
    )
    short_waiting = threading.Thread(
        target=lambda: results.setdefault("short", sched.add_req(_mock_request("short", num_inference_steps=1))),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    long_waiting.start()
    short_waiting.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    long_waiting.join(5)
    short_waiting.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "short", "long"]
    assert results["short"].metrics["scheduler_policy"] == "sjf"
    assert results["short"].metrics["queue_reorder_count"] == 1
    assert results["short"].metrics["estimated_cost_s"] < results["long"].metrics["estimated_cost_s"]


def test_stage1_scheduler_sjf_uses_profile_runtime_estimation(tmp_path):
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "instance_type": "profile-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 10,
                        "latency_s": 5.0,
                    },
                    {
                        "instance_type": "profile-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 50,
                        "latency_s": 0.2,
                    },
                ]
            }
        )
    )
    sched, req_q, res_q = _make_stage1_scheduler(
        policy="sjf",
        profile_path=str(profile_path),
        profile_name="profile-a",
    )
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(target=lambda: sched.add_req(_mock_request("active", num_inference_steps=1)), daemon=True)
    short_profiled = threading.Thread(
        target=lambda: sched.add_req(_mock_request("profile-short", num_inference_steps=50)),
        daemon=True,
    )
    long_profiled = threading.Thread(
        target=lambda: sched.add_req(_mock_request("profile-long", num_inference_steps=10)),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    long_profiled.start()
    short_profiled.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    long_profiled.join(5)
    short_profiled.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "profile-short", "profile-long"]


def test_stage1_scheduler_estimate_cost_counts_num_outputs_once_without_profile():
    sched, _, _ = _make_stage1_scheduler(policy="sjf")

    single_output_cost = sched._estimate_cost_seconds(  # noqa: SLF001
        _mock_request("single", num_inference_steps=10, num_outputs_per_prompt=1)
    )
    multi_output_cost = sched._estimate_cost_seconds(  # noqa: SLF001
        _mock_request("multi", num_inference_steps=10, num_outputs_per_prompt=2)
    )

    assert single_output_cost == pytest.approx(10.0)
    assert multi_output_cost == pytest.approx(20.0)
    assert multi_output_cost == pytest.approx(single_output_cost * 2.0)


def test_stage1_scheduler_scales_injected_estimated_cost_by_remaining_steps():
    sched, _, _ = _make_stage1_scheduler(policy="sjf")
    req = _mock_request(
        "chunked",
        num_inference_steps=20,
        extra_args={"estimated_cost_s": 5.0},
    )

    initial_cost = sched._estimate_cost_seconds(req)  # noqa: SLF001
    req.executed_steps = 10
    resumed_cost = sched._estimate_cost_seconds(req)  # noqa: SLF001

    assert initial_cost == pytest.approx(5.0)
    assert resumed_cost == pytest.approx(2.5)


def test_stage1_scheduler_caches_estimated_cost_for_waiting_plan():
    sched, _, _ = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    original_estimate = sched._estimate_cost_seconds  # noqa: SLF001
    call_count = 0

    def _counting_estimate(request):
        nonlocal call_count
        call_count += 1
        return original_estimate(request)

    sched._estimate_cost_seconds = _counting_estimate  # type: ignore[method-assign]  # noqa: SLF001

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req-1", num_inference_steps=2, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req-2", num_inference_steps=3, extra_args={"slo_ms": 5000.0})
        )
        waiting_requests = list(sched._waiting_queue)  # noqa: SLF001

    sched._build_waiting_plan(waiting_requests, now=time.monotonic())  # noqa: SLF001
    assert call_count == 2

    sched._build_waiting_plan(waiting_requests, now=time.monotonic())  # noqa: SLF001
    sched._build_sjf_queue(waiting_requests, now=time.monotonic())  # noqa: SLF001
    assert call_count == 2


def test_stage1_scheduler_best_effort_aging_uses_request_arrival_time():
    sched, _, _ = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0, aging_factor=1.0)
    older = _mock_request("older", num_inference_steps=8, extra_args={"slo_ms": 5000.0})
    newer = _mock_request("newer", num_inference_steps=7, extra_args={"slo_ms": 5000.0})
    older.arrival_time = 10.0
    newer.arrival_time = 18.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.best_effort_queue] == ["older", "newer"]


@pytest.mark.parametrize("policy", ["slack_age", "slack_cost_age"])
def test_stage1_scheduler_deadline_aware_policies_report_policy_name(policy: str):
    sched, _req_q, _res_q = _make_stage1_scheduler(policy=policy, slo_target_ms=5000.0, aging_factor=1.0)

    with sched._queue_cv:  # noqa: SLF001
        queued = sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req", num_inference_steps=2, extra_args={"slo_ms": 5000.0, "estimated_cost_s": 2.0})
        )

    assert queued.schedule_metrics["scheduler_policy"] == policy


@pytest.mark.parametrize(
    ("method_name", "expected_state"),
    [("finish_request", "finished"), ("fail_request", "failed")],
)
def test_stage1_scheduler_request_terminal_state_updates_hold_queue_lock(method_name: str, expected_state: str):
    sched, _, _ = _make_stage1_scheduler()
    req = _mock_request("req-terminal")
    finished = threading.Event()

    with sched._queue_cv:  # noqa: SLF001
        worker = threading.Thread(target=lambda: (getattr(sched, method_name)(req), finished.set()), daemon=True)
        worker.start()
        time.sleep(0.05)
        assert finished.is_set() is False

    worker.join(5)

    assert finished.is_set() is True
    assert getattr(req, "request_state") == expected_state


def test_stage1_scheduler_reports_waiting_queue_len_and_load():
    sched, req_q, res_q = _make_stage1_scheduler()
    release_first = threading.Event()
    first_enqueued = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        assert first["args"][0].request_ids[0] == "active"
        first_enqueued.set()
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        second = req_q.get(timeout=5)
        assert second["args"][0].request_ids[0] == "waiting"
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="waiting"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(target=lambda: sched.add_req(_mock_request("active")), daemon=True)
    waiting = threading.Thread(target=lambda: sched.add_req(_mock_request("waiting")), daemon=True)

    active.start()
    first_enqueued.wait(timeout=5)
    waiting.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if sched.estimate_waiting_queue_len() == 1:
            break
        time.sleep(0.01)

    assert sched.estimate_waiting_queue_len() == 1
    assert sched.estimate_scheduler_load() == {
        "waiting_queue_len": 1,
        "active_request_count": 1,
        "paused_context_count": 0,
    }

    release_first.set()
    active.join(5)
    waiting.join(5)
    worker.join(5)


def test_stage1_scheduler_abort_removes_waiting_request():
    sched, req_q, res_q = _make_stage1_scheduler()
    release_first = threading.Event()
    first_enqueued = threading.Event()
    results: dict[str, DiffusionOutput] = {}

    def _worker():
        first = req_q.get(timeout=5)
        assert first["args"][0].request_ids[0] == "active"
        first_enqueued.set()
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(
        target=lambda: results.setdefault("active", sched.add_req(_mock_request("active"))),
        daemon=True,
    )
    waiting = threading.Thread(
        target=lambda: results.setdefault("waiting", sched.add_req(_mock_request("waiting"))),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    waiting.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if sched.estimate_waiting_queue_len() == 1:
            break
        time.sleep(0.01)

    assert sched.abort_request("waiting") is True
    release_first.set()

    active.join(5)
    waiting.join(5)
    worker.join(5)

    assert results["waiting"].error_code == "REQUEST_ABORTED"
    assert results["waiting"].error == "Request aborted before dispatch"
    assert sched.estimate_waiting_queue_len() == 0


def test_stage1_scheduler_p95_first_reports_policy_name_without_affecting_existing_policies():
    sched, req_q, res_q = _make_stage1_scheduler(policy="p95-first")
    req = _mock_request("req-p95-first")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), request_id="req-p95-first"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.metrics["scheduler_policy"] == "p95-first"


def test_stage1_scheduler_p95_first_deadline_reports_policy_name_without_affecting_existing_policies():
    sched, req_q, res_q = _make_stage1_scheduler(policy="p95-first-deadline")
    req = _mock_request("req-p95-first-deadline")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), request_id="req-p95-first-deadline"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.metrics["scheduler_policy"] == "p95-first-deadline"


def test_stage1_scheduler_p95_bucket_sjf_normalized_reports_policy_name_without_affecting_existing_policies():
    sched, req_q, res_q = _make_stage1_scheduler(policy="p95-bucket-sjf-normalized")
    req = _mock_request("req-p95-bucket-sjf-normalized")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), request_id="req-p95-bucket-sjf-normalized"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.metrics["scheduler_policy"] == "p95-bucket-sjf-normalized"


def test_stage1_scheduler_p95_first_dynamic_p95_grows_with_backlog():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched.od_config.instance_scheduler_p95_first_backlog_alpha = 1.0

    with sched._queue_cv:
        q1 = sched._enqueue_request_locked(
            _mock_request("req-1", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        dynamic_p95_single, backlog_single, _learned_single, _adjusted_single = sched._compute_dynamic_p95_ms(
            [q1], now=q1.enqueue_time
        )
        q2 = sched._enqueue_request_locked(
            _mock_request("req-2", num_inference_steps=1, extra_args={"estimated_cost_s": 3.0})
        )
        waiting = list(sched._waiting_queue)
        dynamic_p95_double, backlog_double, _learned_double, _adjusted_double = sched._compute_dynamic_p95_ms(
            waiting, now=q2.enqueue_time
        )

    assert backlog_double > backlog_single
    assert dynamic_p95_double > dynamic_p95_single



def test_stage1_scheduler_p95_first_reorders_waiting_queue_by_single_queue_priority():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched.od_config.instance_scheduler_p95_first_size_bias = 5.0

    with sched._queue_cv:
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("long", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("short", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        ordered = [queued.request.request_ids[0] for queued in sched._waiting_queue]  # noqa: SLF001

    assert ordered == ["short", "long"]



def test_stage1_scheduler_p95_first_starvation_boost_promotes_old_request():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched.od_config.instance_scheduler_p95_first_size_bias = 20.0
    sched.od_config.instance_scheduler_p95_first_starvation_threshold_s = 5.0
    sched.od_config.instance_scheduler_p95_first_starvation_boost = 200.0

    old_req = _mock_request("old", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    new_req = _mock_request("new", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    old_req.arrival_time = 0.0
    new_req.arrival_time = 9.9

    with sched._queue_cv:
        sched._enqueue_request_locked(old_req)  # noqa: SLF001
        sched._enqueue_request_locked(new_req)  # noqa: SLF001
        ordered, metrics_by_sequence = sched._build_p95_first_queue(list(sched._waiting_queue), now=10.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in ordered] == ["old", "new"]
    old_metrics = metrics_by_sequence[ordered[0].sequence_id]
    assert old_metrics["starvation_boost"] == pytest.approx(200.0)


def test_stage1_scheduler_p95_first_separates_active_total_backlog_from_chunk_blocking():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched.od_config.instance_scheduler_p95_first_backlog_alpha = 1.0

    active_req = _mock_request("active", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    waiting_req = _mock_request("waiting", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    active_req.arrival_time = 100.0
    waiting_req.arrival_time = 100.5
    active_req.max_steps_this_turn = 2

    active_queued = _QueuedRequest(
        request=active_req,
        enqueue_time=100.0,
        sequence_id=1,
        estimated_cost_s=10.0,
    )
    waiting_queued = _QueuedRequest(
        request=waiting_req,
        enqueue_time=100.5,
        sequence_id=2,
        estimated_cost_s=1.0,
    )

    sched._active_request = active_queued
    sched._active_started_at = 100.0

    total_remaining = sched._active_total_remaining_cost_seconds(now=100.5)
    chunk_remaining = sched._active_chunk_remaining_cost_seconds(now=100.5)
    _dynamic_p95_ms, backlog_s, _learned_p95_ms, _adjusted_p95_ms = sched._compute_dynamic_p95_ms(
        [waiting_queued], now=100.5
    )

    assert total_remaining == pytest.approx(9.5)
    assert chunk_remaining == pytest.approx(1.5)
    assert backlog_s == pytest.approx(10.5)
    assert backlog_s > chunk_remaining + waiting_queued.estimated_cost_s



def test_stage1_scheduler_p95_first_uses_active_chunk_blocking_for_cursor():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")

    active_req = _mock_request("active", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    waiting_req = _mock_request("waiting", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    active_req.arrival_time = 100.0
    waiting_req.arrival_time = 100.5
    active_req.max_steps_this_turn = 2

    active_queued = _QueuedRequest(
        request=active_req,
        enqueue_time=100.0,
        sequence_id=1,
        estimated_cost_s=10.0,
    )
    waiting_queued = _QueuedRequest(
        request=waiting_req,
        enqueue_time=100.5,
        sequence_id=2,
        estimated_cost_s=1.0,
    )

    sched._active_request = active_queued
    sched._active_started_at = 100.0

    ordered, metrics_by_sequence = sched._build_p95_first_queue([waiting_queued], now=100.5)

    assert [queued.request.request_ids[0] for queued in ordered] == ["waiting"]
    waiting_metrics = metrics_by_sequence[waiting_queued.sequence_id]
    assert waiting_metrics["active_chunk_blocking_s"] == pytest.approx(1.5)
    assert waiting_metrics["instance_backlog_total_s"] == pytest.approx(10.5)
    assert waiting_metrics["predicted_finish_latency_ms"] == pytest.approx(2500.0)


def test_stage1_scheduler_p95_first_ignores_legacy_slo_target_without_explicit_base():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first", slo_target_ms=5000.0)
    sched.od_config.instance_scheduler_p95_first_backlog_alpha = 0.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(
            _mock_request("req", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        dynamic_p95_ms, _backlog_s, learned_p95_ms, backlog_adjusted_p95_ms = sched._compute_dynamic_p95_ms(
            [queued], now=queued.enqueue_time
        )

    assert learned_p95_ms == pytest.approx(1000.0)
    assert backlog_adjusted_p95_ms == pytest.approx(0.0)
    assert dynamic_p95_ms == pytest.approx(1000.0)



def test_stage1_scheduler_p95_first_uses_explicit_base_without_reusing_legacy_slo_target():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first", slo_target_ms=5000.0)
    sched.od_config.instance_scheduler_p95_first_base_ms = 2200.0
    sched.od_config.instance_scheduler_p95_first_backlog_alpha = 0.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(
            _mock_request("req", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        dynamic_p95_ms, _backlog_s, learned_p95_ms, backlog_adjusted_p95_ms = sched._compute_dynamic_p95_ms(
            [queued], now=queued.enqueue_time
        )

    assert learned_p95_ms == pytest.approx(2200.0)
    assert backlog_adjusted_p95_ms == pytest.approx(2200.0)
    assert dynamic_p95_ms == pytest.approx(2200.0)


def test_stage1_scheduler_p95_first_learned_p95_uses_max_for_small_history():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched._p95_first_latency_history_ms.extend([100.0, 400.0, 250.0])

    assert sched._learned_p95_ms() == pytest.approx(400.0)



def test_stage1_scheduler_p95_first_learned_p95_uses_nearest_rank_for_larger_history():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched._p95_first_latency_history_ms.extend(float(i * 100) for i in range(1, 21))

    assert sched._learned_p95_ms() == pytest.approx(1900.0)



def test_stage1_scheduler_p95_first_updates_service_rate_from_actual_chunk_runtime():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")

    req = _mock_request("runtime", num_inference_steps=4, extra_args={"estimated_cost_s": 4.0})
    chunk_output = SimpleNamespace(error=None, finished=False, metrics={"executed_steps": 2})
    chunk_work_units = sched._completed_chunk_work_units(req, chunk_output, previous_executed_steps=0)
    sched._record_p95_first_execute_sample(400.0, chunk_work_units)

    waiting_req = _mock_request("waiting", num_inference_steps=3, extra_args={"estimated_cost_s": 9.0})
    queued = _QueuedRequest(request=waiting_req, enqueue_time=0.0, sequence_id=1, estimated_cost_s=9.0)

    assert sched._p95_first_observed_service_ms_per_work_unit == pytest.approx(200.0)
    assert sched._p95_first_estimated_service_ms(queued) == pytest.approx(600.0)



def test_stage1_scheduler_p95_first_learns_slowdown_from_request_latency_over_actual_execute_time():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")

    req = _mock_request("done", num_inference_steps=4, extra_args={"estimated_cost_s": 4.0})
    setattr(req, "_p95_first_cumulative_execute_ms", 400.0)

    sched._record_p95_first_latency_ms(1000.0, request=req)

    assert sched._learned_p95_first_slowdown() == pytest.approx(2.5)



def test_stage1_scheduler_p95_first_reorders_by_normalized_tail_pressure_without_dynamic_p95_knobs():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first")
    sched.od_config.instance_scheduler_p95_first_base_ms = 999999.0
    sched.od_config.instance_scheduler_p95_first_min_ms = 888888.0
    sched.od_config.instance_scheduler_p95_first_max_ms = 999999.0
    sched.od_config.instance_scheduler_p95_first_backlog_alpha = 123.0
    sched._p95_first_observed_service_ms_per_work_unit = 1000.0
    sched._p95_first_slowdown_history.append(2.0)

    older = _mock_request("older", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    older.arrival_time = 0.0
    newer.arrival_time = 19.0

    with sched._queue_cv:
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        ordered, metrics_by_sequence = sched._build_p95_first_queue(list(sched._waiting_queue), now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in ordered] == ["older", "newer"]
    first_metrics = metrics_by_sequence[ordered[0].sequence_id]
    assert first_metrics["learned_slowdown_p95"] == pytest.approx(2.0)
    assert first_metrics["target_latency_ms"] == pytest.approx(20000.0)
    assert first_metrics["service_rate_source"] == "observed_runtime"
    assert "dynamic_p95_ms" not in first_metrics

def test_stage1_scheduler_p95_first_deadline_reorders_by_normalized_synthetic_deadline():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first-deadline")
    sched._p95_first_observed_service_ms_per_work_unit = 1000.0
    sched._p95_first_slowdown_history.append(2.0)

    older = _mock_request("older", num_inference_steps=10, extra_args={"estimated_cost_s": 10.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    older.arrival_time = 0.0
    newer.arrival_time = 19.0

    with sched._queue_cv:
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        ordered, metrics_by_sequence = sched._build_p95_first_deadline_queue(list(sched._waiting_queue), now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in ordered] == ["older", "newer"]
    first_metrics = metrics_by_sequence[ordered[0].sequence_id]
    assert first_metrics["learned_slowdown_p95"] == pytest.approx(2.0)
    assert first_metrics["target_latency_ms"] == pytest.approx(20000.0)
    assert first_metrics["synthetic_deadline_ts"] == pytest.approx(20.0)
    assert first_metrics["slack_s"] == pytest.approx(-10.0)
    assert first_metrics["service_rate_source"] == "observed_runtime"
    assert "dynamic_p95_ms" not in first_metrics



def test_stage1_scheduler_p95_first_deadline_learns_slowdown_from_request_latency_over_actual_execute_time():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-first-deadline")

    req = _mock_request("done", num_inference_steps=4, extra_args={"estimated_cost_s": 4.0})
    setattr(req, "_p95_first_cumulative_execute_ms", 400.0)

    sched._record_p95_first_latency_ms(1000.0, request=req)

    assert sched._learned_p95_first_slowdown() == pytest.approx(2.5)


def test_stage1_scheduler_p95_bucket_sjf_normalized_uses_normalized_target_not_absolute_history():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf-normalized")
    sched._p95_first_latency_history_ms.extend([999999.0])
    sched._p95_first_observed_service_ms_per_work_unit = 1000.0
    sched._p95_first_slowdown_history.append(2.0)

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(
            _mock_request("req", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_normalized_queue(list(sched._waiting_queue), now=queued.enqueue_time)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["req"]
    metrics = metrics_by_sequence[queued.sequence_id]
    assert metrics["learned_slowdown_p95"] == pytest.approx(2.0)
    assert metrics["estimated_service_ms"] == pytest.approx(1000.0)
    assert metrics["target_latency_ms"] == pytest.approx(2000.0)
    assert "history_p95_ms" not in metrics



def test_stage1_scheduler_p95_bucket_sjf_normalized_reorders_by_bucket_then_sjf():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf-normalized")
    sched.od_config.instance_scheduler_p95_bucket_count = 4
    sched.od_config.instance_scheduler_p95_bucket_min_window_ms = 1000.0
    sched._p95_first_observed_service_ms_per_work_unit = 1000.0
    sched._p95_first_slowdown_history.append(2.0)

    urgent_long = _mock_request("urgent-long", num_inference_steps=8, extra_args={"estimated_cost_s": 8.0})
    urgent_short = _mock_request("urgent-short", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    relaxed = _mock_request("relaxed", num_inference_steps=8, extra_args={"estimated_cost_s": 8.0})
    urgent_long.arrival_time = 0.0
    urgent_short.arrival_time = 12.0
    relaxed.arrival_time = 13.0

    with sched._queue_cv:
        q1 = sched._enqueue_request_locked(urgent_long)
        q2 = sched._enqueue_request_locked(urgent_short)
        q3 = sched._enqueue_request_locked(relaxed)
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_normalized_queue([q1, q2, q3], now=13.0)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["urgent-short", "urgent-long", "relaxed"]
    assert metrics_by_sequence[q1.sequence_id]["effective_bucket_id"] == metrics_by_sequence[q2.sequence_id]["effective_bucket_id"]
    assert metrics_by_sequence[q3.sequence_id]["effective_bucket_id"] > metrics_by_sequence[q2.sequence_id]["effective_bucket_id"]


def test_stage1_scheduler_p95_bucket_sjf_normalized_learns_slowdown_from_request_latency_over_actual_execute_time():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf-normalized")

    req = _mock_request("done", num_inference_steps=4, extra_args={"estimated_cost_s": 4.0})
    setattr(req, "_p95_first_cumulative_execute_ms", 400.0)

    sched._record_p95_first_latency_ms(1000.0, request=req)

    assert sched._learned_p95_first_slowdown() == pytest.approx(2.5)


def test_stage1_scheduler_p95_bucket_sjf_target_p95_uses_max_of_history_and_cost():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf")
    sched._p95_first_latency_history_ms.extend([1200.0, 1600.0, 1400.0])

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(
            _mock_request("req", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
        )
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_queue(list(sched._waiting_queue), now=queued.enqueue_time)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["req"]
    metrics = metrics_by_sequence[queued.sequence_id]
    assert metrics["history_p95_ms"] == pytest.approx(1600.0)
    assert metrics["target_p95_ms"] == pytest.approx(1600.0)


def test_stage1_scheduler_p95_bucket_sjf_reorders_by_bucket_then_sjf():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf")
    sched._p95_first_latency_history_ms.extend([10000.0])

    urgent_long = _mock_request("urgent-long", num_inference_steps=8, extra_args={"estimated_cost_s": 8.0})
    urgent_short = _mock_request("urgent-short", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    relaxed = _mock_request("relaxed", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    urgent_long.arrival_time = 0.0
    urgent_short.arrival_time = 0.0
    relaxed.arrival_time = 8.0

    with sched._queue_cv:
        q1 = sched._enqueue_request_locked(urgent_long)
        q2 = sched._enqueue_request_locked(urgent_short)
        q3 = sched._enqueue_request_locked(relaxed)
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_queue([q1, q2, q3], now=9.0)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["urgent-short", "urgent-long", "relaxed"]
    assert metrics_by_sequence[q1.sequence_id]["effective_bucket_id"] == metrics_by_sequence[q2.sequence_id]["effective_bucket_id"]
    assert metrics_by_sequence[q3.sequence_id]["effective_bucket_id"] > metrics_by_sequence[q2.sequence_id]["effective_bucket_id"]


def test_stage1_scheduler_p95_bucket_sjf_negative_urgency_falls_into_bucket_zero():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf")
    sched._p95_first_latency_history_ms.extend([1000.0])

    req = _mock_request("late", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    req.arrival_time = 0.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(req)
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_queue([queued], now=5.0)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["late"]
    assert metrics_by_sequence[queued.sequence_id]["raw_bucket_id"] == 0
    assert metrics_by_sequence[queued.sequence_id]["effective_bucket_id"] == 0


def test_stage1_scheduler_p95_bucket_sjf_starvation_promotion_moves_request_forward():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="p95-bucket-sjf")
    sched.od_config.instance_scheduler_p95_bucket_count = 4
    sched.od_config.instance_scheduler_p95_bucket_min_window_ms = 1000.0
    sched.od_config.instance_scheduler_p95_bucket_starvation_threshold_s = 5.0
    sched.od_config.instance_scheduler_p95_bucket_starvation_promote_levels = 2
    sched._p95_first_latency_history_ms.extend([10000.0])

    old_req = _mock_request("old", num_inference_steps=1, extra_args={"estimated_cost_s": 1.0})
    old_req.arrival_time = 0.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(old_req)
        ordered, metrics_by_sequence = sched._build_p95_bucket_sjf_queue([queued], now=7.0)  # noqa: SLF001

    assert [item.request.request_ids[0] for item in ordered] == ["old"]
    metrics = metrics_by_sequence[queued.sequence_id]
    assert metrics["starvation_promoted"] == 1
    assert metrics["effective_bucket_id"] == 0
    assert metrics["raw_bucket_id"] > metrics["effective_bucket_id"]
