#!/usr/bin/env python3
"""Validate simulation against real_execution benchmark data.

For each (algorithm, rps) in real_execution/, run the simulation using:
  - qwen_A100.json profile (pure hardware time from fcfs baseline)
  - The exact request sequence from the trace file
  - Matching scheduler policy and chunk preemption parameters

Then compare simulation output with real execution CSV and JSON.
"""

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
    "fcfs": {
        "policy": "fcfs",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "sjf_aging": {
        "policy": "sjf_aging",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "sjf_aging_guarded": {
        "policy": "sjf_aging_guarded",
        "chunk_budget": 5,
        "threshold_ms": 6000.0,
    },
    "sjf_aging_no_preempt": {
        "policy": "sjf_aging_no_preempt",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "size_bucket_sjf_aging": {
        "policy": "size_bucket_sjf_aging",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "size_bucket_sjf_aging_no_preempt": {
        "policy": "size_bucket_sjf_aging_no_preempt",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "p95_first": {
        "policy": "p95-first",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "p95_bucket_sjf": {
        "policy": "p95-bucket-sjf",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
    "p95_bucket_sjf_norm": {
        "policy": "p95-bucket-sjf-normalized",
        "chunk_budget": 12,
        "threshold_ms": 12000.0,
    },
}

WORKER = {"sp": 1, "cfg": 1, "tp": 1, "instance_type": "sp1_cfg1_tp1"}


def parse_trace(trace_path: Path) -> list[dict]:
    """Parse trace file, return list of request dicts (no arrival_time yet)."""
    requests = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            r = _parse_sd3_trace_line(line)
            if r is not None:
                requests.append(r)
    return requests


def build_requests_from_trace(trace_requests: list[dict], rps: float) -> list[dict]:
    """Assign arrival times and request IDs based on rps."""
    requests = []
    for i, base in enumerate(trace_requests):
        req = base.copy()
        req["request_id"] = f"req_{i}"
        req["arrival_time"] = i / rps if rps > 0 else 0.0
        requests.append(req)
    return requests


def load_real_json(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_real_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def rps_from_label(label: str) -> float:
    return float(label.replace("p", "."))


def discover_runs(algo_name: str) -> list[tuple[float, Path, Path, Path]]:
    """Return [(rps, trace_path, json_path, csv_path), ...] for an algorithm."""
    algo_dir = REAL_EXEC_DIR / algo_name
    runs = []
    for trace_file in sorted(algo_dir.glob(f"{algo_name}_rps_*.trace.txt")):
        m = re.search(r"_rps_(\d+p\d+)", trace_file.name)
        if not m:
            continue
        rps_label = m.group(1)
        rps_val = rps_from_label(rps_label)
        json_path = algo_dir / f"{algo_name}_rps_{rps_label}.json"
        csv_path = algo_dir / f"{algo_name}_rps_{rps_label}_requests.csv"
        if json_path.exists() and csv_path.exists():
            runs.append((rps_val, trace_file, json_path, csv_path))
    return runs


def compare_metrics(sim: dict, real: dict) -> list[dict]:
    """Compare summary metrics, return list of {metric, sim, real, diff, pct}."""
    keys = [
        "duration", "completed_requests", "throughput_qps",
        "latency_mean", "latency_median", "latency_p50", "latency_p95", "latency_p99",
        "waiting_time_mean", "waiting_time_p95", "waiting_time_p99",
        "service_time_mean", "service_time_p95", "service_time_p99",
    ]
    rows = []
    for k in keys:
        sv = sim.get(k, 0.0)
        rv = real.get(k, 0.0)
        diff = sv - rv
        pct = (diff / rv * 100.0) if rv != 0 else 0.0
        rows.append({"metric": k, "sim": round(sv, 4), "real": round(rv, 4),
                      "diff": round(diff, 4), "pct": round(pct, 2)})
    return rows


def compare_request_order(sim_requests: list[dict], real_csv: list[dict]) -> dict:
    """Compare completion order between simulation and real execution.

    real CSV is sorted by finish_time. We sort sim by finish_time too.
    Compare the sequence of request sizes (since request_ids differ).
    """
    sim_sorted = sorted(sim_requests, key=lambda r: r["finish_time"])
    sim_sizes = [r.get("size", "") for r in sim_sorted]
    real_sizes = [r.get("size", "") for r in real_csv]

    n = min(len(sim_sizes), len(real_sizes))
    match_count = sum(1 for i in range(n) if sim_sizes[i] == real_sizes[i])

    first_mismatch = -1
    for i in range(n):
        if sim_sizes[i] != real_sizes[i]:
            first_mismatch = i
            break

    return {
        "total": n,
        "order_match": match_count,
        "order_match_pct": round(match_count / n * 100, 1) if n > 0 else 0,
        "first_mismatch_idx": first_mismatch,
    }


def run_one(algo_name: str, rps: float, trace_path: Path,
            real_json_path: Path, real_csv_path: Path,
            profile_data: dict) -> dict:
    """Run simulation for one (algorithm, rps) and compare with real data."""
    cfg = ALGO_CONFIG[algo_name]
    trace_requests = parse_trace(trace_path)
    requests = build_requests_from_trace(trace_requests, rps)
    n_requests = len(requests)
    t_end = (n_requests / rps) + 1.0 if rps > 0 else 1000.0

    sim_result = run_simulation(
        requests=requests,
        workers=[WORKER],
        profile_data=profile_data,
        rps=rps,
        t_end=t_end,
        algorithm="round_robin",
        instance_scheduler_policy=cfg["policy"],
        chunk_preempt_budget_steps=cfg["chunk_budget"],
        chunk_preempt_small_request_threshold_ms=cfg["threshold_ms"],
        use_heuristic_cost=False,
    )

    real_json = load_real_json(real_json_path)
    real_csv = load_real_csv(real_csv_path)

    metric_cmp = compare_metrics(sim_result, real_json)
    order_cmp = compare_request_order(sim_result.get("requests", []), real_csv)

    return {
        "algo": algo_name,
        "rps": rps,
        "policy": cfg["policy"],
        "n_requests": n_requests,
        "sim_completed": sim_result["completed_requests"],
        "real_completed": real_json["completed_requests"],
        "metrics": metric_cmp,
        "order": order_cmp,
    }


def format_report(results: list[dict]) -> str:
    lines = []
    lines.append("# 仿真 vs 实测 验证报告\n")
    lines.append(f"Profile: `{PROFILE_PATH.name}`\n")
    lines.append(f"实例数: 1 (单 worker)\n")
    lines.append(f"全局调度: round_robin\n\n")

    for r in results:
        lines.append(f"## {r['algo']} (rps={r['rps']}, policy={r['policy']})\n")
        lines.append(f"请求数: {r['n_requests']} | 仿真完成: {r['sim_completed']} | 实测完成: {r['real_completed']}\n")

        # Order comparison
        o = r["order"]
        lines.append(f"\n### 完成顺序对比\n")
        lines.append(f"- 按 finish_time 排序后, 请求类型(size)匹配率: **{o['order_match']}/{o['total']} ({o['order_match_pct']}%)**\n")
        if o["first_mismatch_idx"] >= 0:
            lines.append(f"- 首次不匹配位置: index={o['first_mismatch_idx']}\n")
        else:
            lines.append(f"- 完成顺序完全一致\n")

        # Metrics comparison
        lines.append(f"\n### 指标对比\n")
        lines.append("| 指标 | 仿真 | 实测 | 差值 | 偏差% |\n")
        lines.append("|------|------|------|------|-------|\n")
        for m in r["metrics"]:
            pct_str = f"{m['pct']:+.2f}%"
            lines.append(f"| {m['metric']} | {m['sim']} | {m['real']} | {m['diff']:+.4f} | {pct_str} |\n")
        lines.append("\n")

    # Summary table
    lines.append("## 汇总\n\n")
    lines.append("| 算法 | rps | latency_mean偏差 | latency_p95偏差 | 完成顺序匹配率 |\n")
    lines.append("|------|-----|-----------------|-----------------|---------------|\n")
    for r in results:
        lat_mean = next((m for m in r["metrics"] if m["metric"] == "latency_mean"), {})
        lat_p95 = next((m for m in r["metrics"] if m["metric"] == "latency_p95"), {})
        o = r["order"]
        lines.append(
            f"| {r['algo']} | {r['rps']} "
            f"| {lat_mean.get('pct', 0):+.2f}% "
            f"| {lat_p95.get('pct', 0):+.2f}% "
            f"| {o['order_match_pct']}% |\n"
        )

    return "".join(lines)


def main():
    if not PROFILE_PATH.exists():
        print(f"Profile not found: {PROFILE_PATH}", file=sys.stderr)
        sys.exit(1)

    profile_data = load_profile(PROFILE_PATH)
    results = []

    algos = sorted(d.name for d in REAL_EXEC_DIR.iterdir()
                   if d.is_dir() and d.name in ALGO_CONFIG)

    for algo_name in algos:
        runs = discover_runs(algo_name)
        if not runs:
            print(f"[SKIP] {algo_name}: no trace/json/csv found")
            continue

        for rps_val, trace_path, json_path, csv_path in runs:
            print(f"Running {algo_name} rps={rps_val} ...")
            try:
                result = run_one(algo_name, rps_val, trace_path, json_path, csv_path, profile_data)
                results.append(result)
                lat_mean = next((m for m in result["metrics"] if m["metric"] == "latency_mean"), {})
                print(f"  -> latency_mean: sim={lat_mean.get('sim')}, real={lat_mean.get('real')}, "
                      f"diff={lat_mean.get('pct', 0):+.2f}%, order_match={result['order']['order_match_pct']}%")
            except Exception as e:
                print(f"  [ERROR] {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    report = format_report(results)
    report_path = REAL_EXEC_DIR / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
