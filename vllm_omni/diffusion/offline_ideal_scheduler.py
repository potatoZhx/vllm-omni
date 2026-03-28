from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_EVENT_RE = re.compile(r"\] (?P<event>(?:REQUEST|QUEUE)_[A-Z_]+) ")
_FIELD_RE = re.compile(r"(\w+)=([^ ]+)")
_RELEVANT_EVENTS = {
    "REQUEST_ARRIVED",
    "REQUEST_STARTED",
    "REQUEST_RESUMED",
    "REQUEST_PREEMPTED",
    "REQUEST_COMPLETED",
    "REQUEST_FAILED",
    "REQUEST_ABORTED",
}


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    width: int | None
    height: int | None
    total_steps: int | None
    arrival_ts: float
    arrival_s: float
    service_time_s: float
    completion_ts: float | None
    actual_latency_s: float | None
    source_policy: str | None = None

    @property
    def size(self) -> str | None:
        if self.width is None or self.height is None:
            return None
        return f"{self.height}x{self.width}"


@dataclass(slots=True)
class GreedySelection:
    slo_s: float
    selected: list[RequestRecord]
    dropped: list[RequestRecord]
    accepted_request_ids: set[str]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "None":
        return None
    return int(value)


def _parse_event_line(line: str) -> tuple[str, dict[str, str]] | None:
    match = _EVENT_RE.search(line)
    if match is None:
        return None
    event = match.group("event")
    if event not in _RELEVANT_EVENTS:
        return None
    fields = dict(_FIELD_RE.findall(line[match.end() :]))
    return event, fields


def _finalize_record(request_id: str, state: dict[str, Any]) -> RequestRecord | None:
    arrival_ts = state.get("arrival_ts")
    service_time_s = state.get("service_time_s", 0.0)
    if arrival_ts is None or service_time_s <= 0.0:
        return None
    completion_ts = state.get("completion_ts")
    actual_latency_s = None
    if completion_ts is not None:
        actual_latency_s = float(completion_ts) - float(arrival_ts)
    return RequestRecord(
        request_id=request_id,
        width=state.get("width"),
        height=state.get("height"),
        total_steps=state.get("total_steps"),
        arrival_ts=float(arrival_ts),
        arrival_s=0.0,
        service_time_s=float(service_time_s),
        completion_ts=float(completion_ts) if completion_ts is not None else None,
        actual_latency_s=actual_latency_s,
        source_policy=state.get("policy"),
    )


def parse_completed_requests_from_log(
    log_path: str | Path,
    *,
    skip_missing_request_id: bool = True,
) -> list[RequestRecord]:
    path = Path(log_path)
    states: dict[str, dict[str, Any]] = {}
    completed_ids: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            parsed = _parse_event_line(raw_line)
            if parsed is None:
                continue
            event, fields = parsed
            request_id = fields.get("request_id")
            if not request_id:
                continue
            if skip_missing_request_id and request_id == "<missing-request-id>":
                continue

            state = states.setdefault(
                request_id,
                {
                    "service_time_s": 0.0,
                    "run_start_ts": None,
                },
            )
            width = _to_int(fields.get("width"))
            if width is not None:
                state["width"] = width
            height = _to_int(fields.get("height"))
            if height is not None:
                state["height"] = height
            total_steps = _to_int(fields.get("total_steps"))
            if total_steps is not None:
                state["total_steps"] = total_steps
            if fields.get("policy"):
                state["policy"] = fields.get("policy")

            arrival_ts = _to_float(fields.get("arrival_ts"))
            if arrival_ts is not None and state.get("arrival_ts") is None:
                state["arrival_ts"] = arrival_ts

            if event in {"REQUEST_STARTED", "REQUEST_RESUMED"}:
                dispatch_ts = _to_float(fields.get("last_dispatch_ts"))
                if dispatch_ts is None:
                    dispatch_ts = _to_float(fields.get("first_dispatch_ts"))
                state["run_start_ts"] = dispatch_ts
                if state.get("first_dispatch_ts") is None:
                    state["first_dispatch_ts"] = dispatch_ts
                continue

            if event == "REQUEST_PREEMPTED":
                preempted_ts = _to_float(fields.get("last_preempted_ts"))
                run_start_ts = state.get("run_start_ts")
                if run_start_ts is not None and preempted_ts is not None:
                    state["service_time_s"] += max(preempted_ts - run_start_ts, 0.0)
                state["run_start_ts"] = None
                continue

            if event in {"REQUEST_COMPLETED", "REQUEST_FAILED", "REQUEST_ABORTED"}:
                end_key = {
                    "REQUEST_COMPLETED": "completion_ts",
                    "REQUEST_FAILED": "failure_ts",
                    "REQUEST_ABORTED": "aborted_ts",
                }[event]
                end_ts = _to_float(fields.get(end_key))
                run_start_ts = state.get("run_start_ts")
                if run_start_ts is not None and end_ts is not None:
                    state["service_time_s"] += max(end_ts - run_start_ts, 0.0)
                elif (
                    event == "REQUEST_COMPLETED"
                    and state.get("service_time_s", 0.0) <= 0.0
                    and end_ts is not None
                ):
                    first_dispatch_ts = _to_float(fields.get("first_dispatch_ts"))
                    if first_dispatch_ts is not None:
                        state["service_time_s"] = max(end_ts - first_dispatch_ts, 0.0)
                state["run_start_ts"] = None
                if event == "REQUEST_COMPLETED" and end_ts is not None:
                    state["completion_ts"] = end_ts
                    completed_ids.append(request_id)

    records: list[RequestRecord] = []
    seen: set[str] = set()
    for request_id in completed_ids:
        if request_id in seen:
            continue
        seen.add(request_id)
        record = _finalize_record(request_id, states[request_id])
        if record is not None:
            records.append(record)

    if not records:
        return []

    base_arrival = min(record.arrival_ts for record in records)
    normalized_records = [
        RequestRecord(
            request_id=record.request_id,
            width=record.width,
            height=record.height,
            total_steps=record.total_steps,
            arrival_ts=record.arrival_ts,
            arrival_s=record.arrival_ts - base_arrival,
            service_time_s=record.service_time_s,
            completion_ts=record.completion_ts,
            actual_latency_s=record.actual_latency_s,
            source_policy=record.source_policy,
        )
        for record in records
    ]
    normalized_records.sort(key=lambda item: (item.arrival_s, item.request_id))
    return normalized_records


def latest_start_deadline_s(request: RequestRecord, slo_s: float) -> float:
    return request.arrival_s + slo_s - request.service_time_s


def _schedule_selected_prefix(active: list[RequestRecord], slo_s: float) -> tuple[list[dict[str, float]], bool]:
    rows: list[dict[str, float]] = []
    current_time_s = 0.0
    feasible = True

    for request in active:
        start_time_s = max(current_time_s, request.arrival_s)
        ddl_s = latest_start_deadline_s(request, slo_s)
        finish_time_s = start_time_s + request.service_time_s
        row = {
            "arrival_s": request.arrival_s,
            "start_time_s": start_time_s,
            "finish_time_s": finish_time_s,
            "latest_start_deadline_s": ddl_s,
        }
        rows.append(row)
        if start_time_s > ddl_s + 1e-9:
            feasible = False
        current_time_s = finish_time_s

    return rows, feasible


def greedy_select_by_latest_start_deadline(
    requests: list[RequestRecord],
    slo_s: float,
) -> GreedySelection:
    ordered = sorted(
        requests,
        key=lambda request: (
            latest_start_deadline_s(request, slo_s),
            request.arrival_s,
            request.service_time_s,
            request.request_id,
        ),
    )

    active: list[RequestRecord] = []

    for request in ordered:
        active.append(request)
        while True:
            scheduled_rows, feasible = _schedule_selected_prefix(active, slo_s)
            if feasible:
                break

            largest_request = max(
                active,
                key=lambda item: (item.service_time_s, item.arrival_s, item.request_id),
            )
            active.remove(largest_request)
            if not active:
                break
            if largest_request.request_id == request.request_id:
                break

    accepted_request_ids = {request.request_id for request in active}
    selected = [request for request in ordered if request.request_id in accepted_request_ids]
    dropped = [request for request in ordered if request.request_id not in accepted_request_ids]
    return GreedySelection(
        slo_s=slo_s,
        selected=selected,
        dropped=dropped,
        accepted_request_ids=accepted_request_ids,
    )


def find_min_slo_for_ratio(
    requests: list[RequestRecord],
    *,
    target_ratio: float = 0.95,
    iterations: int = 60,
) -> GreedySelection:
    if not requests:
        raise ValueError("No completed requests were parsed from the log.")
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")

    target_count = math.ceil(len(requests) * target_ratio)
    low = 0.0
    high = max(
        max((request.actual_latency_s or 0.0) for request in requests),
        max(request.arrival_s for request in requests) + sum(request.service_time_s for request in requests),
    )

    best = greedy_select_by_latest_start_deadline(requests, high)
    while len(best.selected) < target_count:
        high = max(high * 2.0, 1e-6)
        best = greedy_select_by_latest_start_deadline(requests, high)

    for _ in range(iterations):
        mid = (low + high) / 2.0
        candidate = greedy_select_by_latest_start_deadline(requests, mid)
        if len(candidate.selected) >= target_count:
            high = mid
            best = candidate
        else:
            low = mid

    return best


def build_schedule_rows(selection: GreedySelection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scheduled_rows, _ = _schedule_selected_prefix(selection.selected, selection.slo_s)

    for order_index, (request, scheduled_row) in enumerate(zip(selection.selected, scheduled_rows), start=1):
        finish_time_s = scheduled_row["finish_time_s"]
        rows.append(
            {
                "order": order_index,
                "status": "selected",
                "request_id": request.request_id,
                "arrival_s": round(request.arrival_s, 6),
                "service_time_s": round(request.service_time_s, 6),
                "latest_start_deadline_s": round(scheduled_row["latest_start_deadline_s"], 6),
                "completion_deadline_s": round(request.arrival_s + selection.slo_s, 6),
                "start_time_s": round(scheduled_row["start_time_s"], 6),
                "finish_time_s": round(finish_time_s, 6),
                "meets_slo_under_sequence": finish_time_s <= request.arrival_s + selection.slo_s + 1e-9,
                "width": request.width,
                "height": request.height,
                "total_steps": request.total_steps,
                "size": request.size,
            }
        )

    for offset, request in enumerate(selection.dropped, start=1):
        rows.append(
            {
                "order": len(selection.selected) + offset,
                "status": "dropped",
                "request_id": request.request_id,
                "arrival_s": round(request.arrival_s, 6),
                "service_time_s": round(request.service_time_s, 6),
                "latest_start_deadline_s": round(latest_start_deadline_s(request, selection.slo_s), 6),
                "completion_deadline_s": round(request.arrival_s + selection.slo_s, 6),
                "start_time_s": None,
                "finish_time_s": None,
                "meets_slo_under_sequence": False,
                "width": request.width,
                "height": request.height,
                "total_steps": request.total_steps,
                "size": request.size,
            }
        )

    return rows


def summarize_selection(
    requests: list[RequestRecord],
    selection: GreedySelection,
    *,
    target_ratio: float,
    log_path: str | Path,
) -> dict[str, Any]:
    target_count = math.ceil(len(requests) * target_ratio)
    rows = build_schedule_rows(selection)
    selected_on_time = sum(1 for row in rows if row["status"] == "selected" and row["meets_slo_under_sequence"])
    return {
        "log_path": str(log_path),
        "total_completed_requests": len(requests),
        "target_ratio": target_ratio,
        "target_count": target_count,
        "selected_count": len(selection.selected),
        "selected_ratio": len(selection.selected) / len(requests),
        "selected_on_time_count": selected_on_time,
        "selected_on_time_ratio": selected_on_time / len(requests),
        "best_slo_s": selection.slo_s,
        "best_slo_ms": selection.slo_s * 1000.0,
        "max_selected_service_time_s": max((request.service_time_s for request in selection.selected), default=0.0),
        "selected_request_ids": [request.request_id for request in selection.selected],
        "dropped_request_ids": [request.request_id for request in selection.dropped],
    }


def write_schedule_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order",
        "status",
        "request_id",
        "arrival_s",
        "service_time_s",
        "latest_start_deadline_s",
        "completion_deadline_s",
        "start_time_s",
        "finish_time_s",
        "meets_slo_under_sequence",
        "width",
        "height",
        "total_steps",
        "size",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse stage1 scheduler logs and compute an offline ideal p95 schedule.")
    parser.add_argument("log_path", help="Path to worker log, for example logs/.../instance_logs/worker0.log")
    parser.add_argument("--target-ratio", type=float, default=0.95, help="Target on-time ratio, default 0.95")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to <log_path>.offline_ideal_p95",
    )
    parser.add_argument(
        "--include-missing-request-id",
        action="store_true",
        help="Include warmup requests logged as <missing-request-id>.",
    )
    args = parser.parse_args(argv)

    requests = parse_completed_requests_from_log(
        args.log_path,
        skip_missing_request_id=not args.include_missing_request_id,
    )
    selection = find_min_slo_for_ratio(
        requests,
        target_ratio=args.target_ratio,
    )
    summary = summarize_selection(
        requests,
        selection,
        target_ratio=args.target_ratio,
        log_path=args.log_path,
    )
    rows = build_schedule_rows(selection)

    prefix = args.output_prefix
    if prefix is None:
        prefix = f"{args.log_path}.offline_ideal_p95"

    summary_path = f"{prefix}.summary.json"
    schedule_path = f"{prefix}.schedule.csv"
    write_summary_json(summary_path, summary)
    write_schedule_csv(schedule_path, rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_json={summary_path}")
    print(f"schedule_csv={schedule_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
