"""
Extract per-request metrics from real_execution worker0.log files.

For each algorithm, produces:
  - Per-RPS CSV (sorted by finish_time) and JSON summary
  - A profile.json with average service_time per request type (4 size classes)
"""

import re
import json
import csv
import os
import statistics
from pathlib import Path

REAL_EXEC_DIR = Path(r"C:\Users\Asus\Desktop\real_execution")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "real_execution"

REQUEST_COMPLETED_RE = re.compile(
    r"REQUEST_COMPLETED "
    r"request_id=(?P<request_id>\S+) "
    r"width=(?P<width>\d+) "
    r"height=(?P<height>\d+) "
    r"total_steps=(?P<total_steps>\d+) "
    r"executed_steps=(?P<executed_steps>\d+) "
    r"remaining_steps=(?P<remaining_steps>\d+) "
    r"dispatch_epoch=(?P<dispatch_epoch>\d+) "
    r"chunk_budget_steps=(?P<chunk_budget_steps>\S+) "
    r"arrival_ts=(?P<arrival_ts>\S+) "
    r"first_enqueue_ts=(?P<first_enqueue_ts>\S+) "
    r"first_dispatch_ts=(?P<first_dispatch_ts>\S+) "
    r"last_dispatch_ts=(?P<last_dispatch_ts>\S+) "
    r"last_preempted_ts=(?P<last_preempted_ts>\S+) "
    r"completion_ts=(?P<completion_ts>\S+) "
    r"failure_ts=(?P<failure_ts>\S+) "
    r"aborted_ts=(?P<aborted_ts>\S+) "
    r"queue_len=(?P<queue_len>\d+) "
    r"latency_ms=(?P<latency_ms>\S+) "
    r"policy=(?P<policy>\S+)"
)

WARMUP_PROMPT_RE = re.compile(
    r"Diffusion chat request (chatcmpl-\w+):.*stage1-warmup"
)

RPS_SPECIAL = {
    "sjf_aging_guarded": [0.125],
}

CSV_FIELDS = [
    "algorithm", "rps", "request_id",
    "arrival_time", "start_time", "finish_time",
    "latency", "waiting_time", "service_time",
    "size", "steps", "executed_steps", "dispatch_epoch",
]


def parse_log(log_path: str):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    warmup_ids = {"<missing-request-id>"}
    for m in WARMUP_PROMPT_RE.finditer(content):
        warmup_ids.add(m.group(1))

    records = []
    for m in REQUEST_COMPLETED_RE.finditer(content):
        d = m.groupdict()
        if d["request_id"] in warmup_ids:
            continue
        if int(d["total_steps"]) == 1:
            continue
        rec = {
            "request_id": d["request_id"],
            "width": int(d["width"]),
            "height": int(d["height"]),
            "total_steps": int(d["total_steps"]),
            "executed_steps": int(d["executed_steps"]),
            "dispatch_epoch": int(d["dispatch_epoch"]),
            "arrival_ts": float(d["arrival_ts"]),
            "first_enqueue_ts": float(d["first_enqueue_ts"]),
            "first_dispatch_ts": float(d["first_dispatch_ts"]),
            "last_dispatch_ts": float(d["last_dispatch_ts"]),
            "completion_ts": float(d["completion_ts"]),
            "latency_ms": float(d["latency_ms"]),
        }
        records.append(rec)

    records.sort(key=lambda r: r["arrival_ts"])
    return records


def split_groups(records, gap_threshold=50.0):
    if not records:
        return []
    groups = [[records[0]]]
    for i in range(1, len(records)):
        if records[i]["arrival_ts"] - records[i - 1]["arrival_ts"] > gap_threshold:
            groups.append([])
        groups[-1].append(records[i])
    return groups


def extract_rps_stream(group, rps_val, target_count=100):
    """From a group of records sorted by arrival_ts, extract exactly
    `target_count` requests whose inter-arrival gaps match the expected
    rps (1/rps seconds).  Handles the case where a group contains
    interleaved requests from a concurrent benchmark run."""
    expected_gap = 1.0 / rps_val
    tolerance = expected_gap * 0.35

    # Fast path: if the last `target_count` already have consistent gaps,
    # return them directly (covers the common 108-request groups).
    last_n = group[-target_count:]
    gaps = [last_n[i + 1]["arrival_ts"] - last_n[i]["arrival_ts"]
            for i in range(len(last_n) - 1)]
    bad = sum(1 for g in gaps if abs(g - expected_gap) > tolerance)
    if bad == 0:
        return last_n

    # Slow path: greedily select requests that form a regular arrival
    # stream at the expected rps, starting from the end and working
    # backwards to prefer later (non-warmup) requests.
    selected = [group[-1]]
    for i in range(len(group) - 2, -1, -1):
        gap = selected[-1]["arrival_ts"] - group[i]["arrival_ts"]
        if abs(gap - expected_gap) <= tolerance:
            selected.append(group[i])
            if len(selected) == target_count:
                break

    if len(selected) == target_count:
        selected.reverse()
        return selected

    # Fallback: return last target_count (shouldn't normally reach here)
    return group[-target_count:]


def discover_rps(algo_dir: str, algo_name: str):
    if algo_name in RPS_SPECIAL:
        return RPS_SPECIAL[algo_name]

    rps_values = []
    for fname in sorted(os.listdir(algo_dir)):
        m = re.match(r"metrics_rps_(\d+)p(\d+)\.json", fname)
        if m:
            rps_values.append(float(f"{m.group(1)}.{m.group(2)}"))
    return sorted(rps_values)


def percentile(data, p):
    n = len(data)
    if n == 0:
        return 0.0
    s = sorted(data)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < n else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


def build_summary(algo, rps_val, recs):
    lats = [r["latency_ms"] / 1000.0 for r in recs]
    waits = [r["first_dispatch_ts"] - r["arrival_ts"] for r in recs]
    services = [r["completion_ts"] - r["first_dispatch_ts"] for r in recs]
    duration = recs[-1]["completion_ts"] - recs[0]["arrival_ts"]

    return {
        "algorithm": algo,
        "rps": rps_val,
        "duration": round(duration, 4),
        "completed_requests": len(recs),
        "failed_requests": 0,
        "throughput_qps": round(len(recs) / duration, 4) if duration > 0 else 0,
        "latency_mean": round(statistics.mean(lats), 4),
        "latency_median": round(statistics.median(lats), 4),
        "latency_p50": round(percentile(lats, 50), 4),
        "latency_p95": round(percentile(lats, 95), 4),
        "latency_p99": round(percentile(lats, 99), 4),
        "waiting_time_mean": round(statistics.mean(waits), 4),
        "waiting_time_p95": round(percentile(waits, 95), 4),
        "waiting_time_p99": round(percentile(waits, 99), 4),
        "service_time_mean": round(statistics.mean(services), 4),
        "service_time_p95": round(percentile(services, 95), 4),
        "service_time_p99": round(percentile(services, 99), 4),
    }


SIZE_TO_STEPS = {
    (512, 512): 20,
    (768, 768): 20,
    (1024, 1024): 25,
    (1536, 1536): 35,
}


def build_profile(all_recs):
    """Average service_time per request type, structured like qwen_A100.json."""
    by_type = {}
    for r in all_recs:
        key = (r["width"], r["height"])
        by_type.setdefault(key, []).append(
            r["completion_ts"] - r["first_dispatch_ts"]
        )

    profiles = []
    for (w, h) in sorted(by_type.keys(), key=lambda k: k[0] * k[1]):
        vals = by_type[(w, h)]
        profiles.append({
            "instance_type": "sp1_cfg1_tp1",
            "task_type": "image",
            "width": w,
            "height": h,
            "num_frames": 1,
            "steps": SIZE_TO_STEPS.get((w, h), int(all_recs[0]["total_steps"])),
            "latency_s": round(statistics.mean(vals), 4),
            "count": len(vals),
            "latency_median": round(statistics.median(vals), 4),
            "latency_min": round(min(vals), 4),
            "latency_max": round(max(vals), 4),
        })
    return {"profiles": profiles}


def rps_label(rps_val):
    return str(rps_val).replace(".", "p")


def write_csv(out_path, algo, rps_val, recs):
    base_ts = recs[0]["arrival_ts"]
    # recs is already sorted by arrival_ts; assign numeric IDs in that order
    by_arrival = sorted(enumerate(recs), key=lambda t: t[1]["arrival_ts"])
    arrival_rank = {id(r): rank for rank, (_, r) in enumerate(by_arrival)}
    rows = []
    for r in recs:
        rows.append({
            "algorithm": algo,
            "rps": rps_val,
            "request_id": arrival_rank[id(r)],
            "arrival_time": round(r["arrival_ts"] - base_ts, 4),
            "start_time": round(r["first_dispatch_ts"] - base_ts, 4),
            "finish_time": round(r["completion_ts"] - base_ts, 4),
            "latency": round(r["latency_ms"] / 1000.0, 4),
            "waiting_time": round(r["first_dispatch_ts"] - r["arrival_ts"], 4),
            "service_time": round(r["completion_ts"] - r["first_dispatch_ts"], 4),
            "size": f"{r['width']}x{r['height']}",
            "steps": r["total_steps"],
            "executed_steps": r["executed_steps"],
            "dispatch_epoch": r["dispatch_epoch"],
        })
    rows.sort(key=lambda x: x["finish_time"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_trace(out_path, rows):
    """Write trace file sorted by arrival_time, using numeric request_id from CSV."""
    by_arrival = sorted(rows, key=lambda x: x["arrival_time"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in by_arrival:
            size = r["size"]
            w, h = size.split("x")
            f.write(
                f"Request(request_id={r['request_id']}, timestamp=0.0, "
                f"height={h}, width={w}, num_frames=1, "
                f"prompt='Random prompt {r['request_id']} for benchmarking "
                f"diffusion models', num_inference_steps={r['steps']})\n"
            )


def process_algorithm(algo_name):
    algo_src = REAL_EXEC_DIR / algo_name
    log_path = algo_src / "instance_logs" / "worker0.log"
    if not log_path.exists():
        print(f"  [SKIP] {algo_name}: worker0.log not found")
        return

    rps_values = discover_rps(str(algo_src), algo_name)
    if not rps_values:
        print(f"  [SKIP] {algo_name}: no rps discovered")
        return

    records = parse_log(str(log_path))
    print(f"  Parsed {len(records)} non-warmup REQUEST_COMPLETED")

    groups = split_groups(records)
    print(f"  Found {len(groups)} groups (need {len(rps_values)} rps)")

    viable = [g for g in groups if len(g) >= 100]
    if len(viable) < len(rps_values):
        print(f"  [WARN] only {len(viable)} viable groups (>=100 reqs) "
              f"for {len(rps_values)} rps levels")
        rps_values = rps_values[: len(viable)]
    groups = viable[-len(rps_values):]

    out_dir = OUTPUT_DIR / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_real_recs = []
    json_summaries = []

    for group, rps_val in zip(groups, rps_values):
        real = extract_rps_stream(group, rps_val, target_count=100)
        all_real_recs.extend(real)

        rlabel = rps_label(rps_val)
        csv_path = out_dir / f"{algo_name}_rps_{rlabel}_requests.csv"
        csv_rows = write_csv(str(csv_path), algo_name, rps_val, real)

        trace_path = out_dir / f"{algo_name}_rps_{rlabel}.trace.txt"
        write_trace(str(trace_path), csv_rows)

        summary = build_summary(algo_name, rps_val, real)
        json_summaries.append(summary)

        json_path = out_dir / f"{algo_name}_rps_{rlabel}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  rps={rps_val}: {len(real)} requests, "
              f"mean_lat={summary['latency_mean']:.3f}s, "
              f"duration={summary['duration']:.1f}s -> {csv_path.name}")

    profile = build_profile(all_real_recs)
    profile_path = out_dir / "profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"  Profile: {len(profile['profiles'])} request types -> {profile_path.name}")


def main():
    print(f"Source: {REAL_EXEC_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for algo_name in sorted(os.listdir(str(REAL_EXEC_DIR))):
        algo_path = REAL_EXEC_DIR / algo_name
        if not algo_path.is_dir():
            continue
        print(f"Processing {algo_name}...")
        process_algorithm(algo_name)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
