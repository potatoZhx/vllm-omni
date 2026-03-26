from __future__ import annotations

import pytest

from vllm_omni.diffusion.offline_ideal_scheduler import (
    RequestRecord,
    find_min_slo_for_ratio,
    greedy_select_by_latest_start_deadline,
    parse_completed_requests_from_log,
)


def test_parse_completed_requests_from_log_accumulates_preempted_service(tmp_path):
    log_path = tmp_path / "worker0.log"
    log_path.write_text(
        "\n".join(
            [
                "[Stage-0] INFO 03-25 17:28:29 [stage1_scheduler.py:175] REQUEST_ARRIVED request_id=req-a width=512 height=512 total_steps=20 arrival_ts=10.0 first_enqueue_ts=10.1 first_dispatch_ts=None last_dispatch_ts=None last_preempted_ts=None completion_ts=None failure_ts=None aborted_ts=None queue_len=1 latency_ms=None policy=fcfs",
                "[Stage-0] INFO 03-25 17:28:29 [stage1_scheduler.py:175] REQUEST_STARTED request_id=req-a width=512 height=512 total_steps=20 arrival_ts=10.0 first_enqueue_ts=10.1 first_dispatch_ts=11.0 last_dispatch_ts=11.0 last_preempted_ts=None completion_ts=None failure_ts=None aborted_ts=None queue_len=0 latency_ms=None policy=fcfs",
                "[Stage-0] INFO 03-25 17:28:29 [stage1_scheduler.py:175] REQUEST_PREEMPTED request_id=req-a width=512 height=512 total_steps=20 arrival_ts=10.0 first_enqueue_ts=10.1 first_dispatch_ts=11.0 last_dispatch_ts=11.0 last_preempted_ts=12.5 completion_ts=None failure_ts=None aborted_ts=None queue_len=1 latency_ms=2500.0 policy=fcfs",
                "[Stage-0] INFO 03-25 17:28:29 [stage1_scheduler.py:175] REQUEST_RESUMED request_id=req-a width=512 height=512 total_steps=20 arrival_ts=10.0 first_enqueue_ts=10.1 first_dispatch_ts=11.0 last_dispatch_ts=13.0 last_preempted_ts=12.5 completion_ts=None failure_ts=None aborted_ts=None queue_len=0 latency_ms=None policy=fcfs",
                "[Stage-0] INFO 03-25 17:28:29 [stage1_scheduler.py:175] REQUEST_COMPLETED request_id=req-a width=512 height=512 total_steps=20 arrival_ts=10.0 first_enqueue_ts=10.1 first_dispatch_ts=11.0 last_dispatch_ts=13.0 last_preempted_ts=12.5 completion_ts=14.75 failure_ts=None aborted_ts=None queue_len=0 latency_ms=4750.0 policy=fcfs",
            ]
        ),
        encoding="utf-8",
    )

    requests = parse_completed_requests_from_log(log_path)

    assert len(requests) == 1
    assert requests[0].request_id == "req-a"
    assert requests[0].arrival_s == 0.0
    assert requests[0].service_time_s == pytest.approx(3.25)
    assert requests[0].actual_latency_s == pytest.approx(4.75)


def test_greedy_selection_drops_largest_request_when_latest_start_is_missed():
    requests = [
        RequestRecord("a", 512, 512, 20, 0.0, 0.0, 8.0, 8.0, 8.0),
        RequestRecord("b", 512, 512, 20, 0.0, 0.0, 4.0, 4.0, 4.0),
        RequestRecord("c", 512, 512, 20, 0.0, 0.0, 4.0, 4.0, 4.0),
    ]

    selection = greedy_select_by_latest_start_deadline(requests, 8.0)

    assert [request.request_id for request in selection.selected] == ["b", "c"]
    assert [request.request_id for request in selection.dropped] == ["a"]


def test_find_min_slo_for_ratio_returns_smallest_feasible_value():
    requests = [
        RequestRecord("a", 512, 512, 20, 0.0, 0.0, 8.0, 8.0, 8.0),
        RequestRecord("b", 512, 512, 20, 0.0, 0.0, 4.0, 4.0, 4.0),
        RequestRecord("c", 512, 512, 20, 0.0, 0.0, 4.0, 4.0, 4.0),
    ]

    selection = find_min_slo_for_ratio(requests, target_ratio=2 / 3)

    assert selection.slo_s == pytest.approx(8.0, abs=1e-6)
    assert [request.request_id for request in selection.selected] == ["b", "c"]
