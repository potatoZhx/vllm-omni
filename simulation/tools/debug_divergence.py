#!/usr/bin/env python3
"""Debug request-by-request divergence between simulation and real execution."""

import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import (
    _parse_sd3_trace_line,
    load_profile,
    run_simulation,
)

REAL_EXEC_DIR = Path(__file__).resolve().parent.parent / "real_execution"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "profile" / "qwen_A100.json"

ALGO_CONFIG = {
    "fcfs": {"policy": "fcfs", "chunk_budget": 12, "threshold_ms": 12000.0},
    "sjf_aging": {"policy": "sjf_aging", "chunk_budget": 12, "threshold_ms": 12000.0},
    "sjf_aging_guarded": {"policy": "sjf_aging_guarded", "chunk_budget": 5, "threshold_ms": 6000.0},
    "sjf_aging_no_preempt": {"policy": "sjf_aging_no_preempt", "chunk_budget": 12, "threshold_ms": 12000.0},
    "size_bucket_sjf_aging": {"policy": "size_bucket_sjf_aging", "chunk_budget": 12, "threshold_ms": 12000.0},
    "size_bucket_sjf_aging_no_preempt": {"policy": "size_bucket_sjf_aging_no_preempt", "chunk_budget": 12, "threshold_ms": 12000.0},
    "p95_first": {"policy": "p95-first", "chunk_budget": 12, "threshold_ms": 12000.0},
    "p95_bucket_sjf": {"policy": "p95-bucket-sjf", "chunk_budget": 12, "threshold_ms": 12000.0},
    "p95_bucket_sjf_norm": {"policy": "p95-bucket-sjf-normalized", "chunk_budget": 12, "threshold_ms": 12000.0},
}

WORKER = {"sp": 1, "cfg": 1, "tp": 1, "instance_type": "sp1_cfg1_tp1"}


def parse_trace(trace_path):
    requests = []
    with open(trace_path) as f:
        for line in f:
            r = _parse_sd3_trace_line(line)
            if r:
                requests.append(r)
    return requests


def build_requests_from_trace(trace_requests, rps):
    requests = []
    for i, base in enumerate(trace_requests):
        req = base.copy()
        req["request_id"] = f"req_{i}"
        req["arrival_time"] = i / rps if rps > 0 else 0.0
        requests.append(req)
    return requests


def extract_req_num(req_id):
    if isinstance(req_id, str) and req_id.startswith("req_"):
        return int(req_id[4:])
    return int(req_id)


def run_comparison(algo_name, rps_label):
    rps = float(rps_label.replace("p", "."))
    cfg = ALGO_CONFIG[algo_name]
    algo_dir = REAL_EXEC_DIR / algo_name

    trace_path = algo_dir / f"{algo_name}_rps_{rps_label}.trace.txt"
    csv_path = algo_dir / f"{algo_name}_rps_{rps_label}_requests.csv"
    json_path = algo_dir / f"{algo_name}_rps_{rps_label}.json"

    if not all(p.exists() for p in [trace_path, csv_path, json_path]):
        print(f"Files missing for {algo_name} rps={rps_label}")
        return

    profile_data = load_profile(PROFILE_PATH)
    trace_reqs = parse_trace(trace_path)
    requests = build_requests_from_trace(trace_reqs, rps)
    n = len(requests)
    t_end = (n / rps) + 1.0

    sim_result = run_simulation(
        requests=requests, workers=[WORKER], profile_data=profile_data,
        rps=rps, t_end=t_end, algorithm="round_robin",
        instance_scheduler_policy=cfg["policy"],
        chunk_preempt_budget_steps=cfg["chunk_budget"],
        chunk_preempt_small_request_threshold_ms=cfg["threshold_ms"],
        use_heuristic_cost=False,
    )

    with open(csv_path) as f:
        real_rows = list(csv.DictReader(f))

    sim_reqs = sim_result["requests"]
    sim_by_finish = sorted(sim_reqs, key=lambda r: r["finish_time"])
    real_by_finish = real_rows  # already sorted by finish_time

    # Also sort by request_id for per-request comparison
    sim_by_id = {extract_req_num(r["request_id"]): r for r in sim_reqs}

    print(f"\n{'='*80}")
    print(f"  {algo_name} rps={rps_label} (policy={cfg['policy']})")
    print(f"{'='*80}")

    # 1. Completion order comparison (by request_id)
    sim_finish_order = [extract_req_num(r["request_id"]) for r in sim_by_finish]
    real_finish_order = [int(r["request_id"]) for r in real_by_finish]

    print(f"\n--- Completion order (by request_id) ---")
    n_id_match = sum(1 for a, b in zip(sim_finish_order, real_finish_order) if a == b)
    n_size_match = sum(1 for a, b in zip(sim_by_finish, real_by_finish) if a.get("size") == b.get("size"))
    print(f"ID match: {n_id_match}/100, Size match: {n_size_match}/100")

    # Show first divergences
    print(f"\n--- First divergences in completion order ---")
    print(f"{'Pos':>4} {'Sim_ID':>7} {'Real_ID':>7} {'Sim_Size':>12} {'Real_Size':>12} {'Sim_Finish':>12} {'Real_Finish':>12}")
    div_count = 0
    for i in range(min(len(sim_finish_order), len(real_finish_order))):
        s_id = sim_finish_order[i]
        r_id = real_finish_order[i]
        if s_id != r_id:
            s = sim_by_finish[i]
            r = real_by_finish[i]
            print(f"{i:>4} {s_id:>7} {r_id:>7} {s.get('size',''):>12} {r.get('size',''):>12} {s['finish_time']:>12.3f} {float(r['finish_time']):>12.3f}")
            div_count += 1
            if div_count >= 20:
                print("  ... (truncated)")
                break

    # 2. Per-request timing comparison (sorted by request_id)
    print(f"\n--- Per-request latency comparison (sorted by request_id) ---")
    print(f"{'ReqID':>6} {'Size':>12} {'Sim_Lat':>10} {'Real_Lat':>10} {'Diff':>10} {'Sim_Wait':>10} {'Real_Wait':>10} {'Sim_Finish':>12} {'Real_Finish':>12}")
    large_diffs = []
    for i in range(100):
        s = sim_by_id.get(i)
        r_list = [row for row in real_rows if int(row["request_id"]) == i]
        if not s or not r_list:
            continue
        r = r_list[0]
        s_lat = s["latency"]
        r_lat = float(r["latency"])
        diff = s_lat - r_lat
        s_wait = s["waiting_time"]
        r_wait = float(r["waiting_time"])
        s_fin = s["finish_time"]
        r_fin = float(r["finish_time"])

        if abs(diff) > 2.0:
            large_diffs.append((i, s.get("size"), s_lat, r_lat, diff, s_wait, r_wait, s_fin, r_fin))

    for item in large_diffs[:30]:
        i, size, s_lat, r_lat, diff, s_wait, r_wait, s_fin, r_fin = item
        print(f"{i:>6} {size:>12} {s_lat:>10.3f} {r_lat:>10.3f} {diff:>+10.3f} {s_wait:>10.3f} {r_wait:>10.3f} {s_fin:>12.3f} {r_fin:>12.3f}")
    if len(large_diffs) > 30:
        print(f"  ... ({len(large_diffs)} requests with |diff| > 2s)")
    print(f"Total requests with |latency diff| > 2s: {len(large_diffs)}")

    # 3. Summary stats comparison
    with open(json_path) as f:
        real_json = json.load(f)
    print(f"\n--- Summary metrics ---")
    for key in ["latency_mean", "latency_p95", "latency_p99", "duration", "throughput_qps"]:
        sv = sim_result.get(key, 0)
        rv = real_json.get(key, 0)
        pct = (sv - rv) / rv * 100 if rv else 0
        print(f"  {key}: sim={sv:.4f} real={rv:.4f} diff={pct:+.2f}%")

    # 4. Check chunk behavior for preemption algos
    preempt_reqs = [r for r in sim_reqs if len(r.get("start_times", [])) > 1]
    if preempt_reqs:
        print(f"\n--- Preemption in simulation ---")
        print(f"Preempted requests: {len(preempt_reqs)}")
        sizes_preempted = {}
        for r in preempt_reqs:
            s = r.get("size", "?")
            sizes_preempted[s] = sizes_preempted.get(s, 0) + 1
        print(f"By size: {sizes_preempted}")
        for r in preempt_reqs[:5]:
            n_dispatches = len(r.get("start_times", []))
            chunks = r.get("chunk_service_times", [])
            print(f"  req={r['request_id']} size={r.get('size')} dispatches={n_dispatches} chunks={[round(c,3) for c in chunks]} total_svc={sum(chunks):.3f}")

    real_preempted = [r for r in real_rows if int(r.get("dispatch_epoch", 1)) > 1]
    if real_preempted:
        print(f"\nReal preempted: {len(real_preempted)}")
        for r in real_preempted[:5]:
            print(f"  req={r['request_id']} size={r['size']} epoch={r['dispatch_epoch']} exec_steps={r['executed_steps']} svc_time={r['service_time']}")


if __name__ == "__main__":
    targets = [
        ("sjf_aging_no_preempt", "0p125"),
        ("sjf_aging", "0p125"),
        ("sjf_aging", "0p075"),
        ("p95_bucket_sjf", "0p075"),
        ("p95_bucket_sjf", "0p125"),
        ("size_bucket_sjf_aging", "0p075"),
        ("size_bucket_sjf_aging", "0p125"),
    ]
    for algo, rps in targets:
        run_comparison(algo, rps)
