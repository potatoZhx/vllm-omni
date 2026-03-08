#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate summary_runs.csv by parallel config and request type.")
    parser.add_argument("--summary-csv", required=True, help="Path to summary_runs.csv")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Default: <summary_dir>/summary_agg.csv",
    )
    return parser.parse_args()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else summary_csv.parent / "summary_agg.csv"

    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("num_gpus"),
            row.get("parallel_name"),
            row.get("vae_patch_parallel_size"),
            row.get("tensor_parallel_size"),
            row.get("ulysses_degree"),
            row.get("ring_degree"),
            row.get("cfg_parallel_size"),
            row.get("request_type_id"),
            row.get("height"),
            row.get("width"),
            row.get("num_frames"),
            row.get("num_inference_steps"),
        )
        groups[key].append(row)

    out_rows = []
    for key, g in sorted(groups.items()):
        latency_seconds_list = [safe_float(x.get("latency_seconds", "0")) for x in g]
        latency_ms_list = [safe_float(x.get("latency_ms", "0")) for x in g]

        out_rows.append(
            {
                "num_gpus": key[0],
                "parallel_name": key[1],
                "vae_patch_parallel_size": key[2],
                "tensor_parallel_size": key[3],
                "ulysses_degree": key[4],
                "ring_degree": key[5],
                "cfg_parallel_size": key[6],
                "request_type_id": key[7],
                "height": key[8],
                "width": key[9],
                "num_frames": key[10],
                "num_inference_steps": key[11],
                "repeat_count": len(g),
                "latency_seconds_avg": sum(latency_seconds_list) / len(latency_seconds_list),
                "latency_seconds_min": min(latency_seconds_list),
                "latency_seconds_max": max(latency_seconds_list),
                "latency_ms_avg": sum(latency_ms_list) / len(latency_ms_list),
                "latency_ms_min": min(latency_ms_list),
                "latency_ms_max": max(latency_ms_list),
            }
        )

    fields = [
        "num_gpus",
        "parallel_name",
        "vae_patch_parallel_size",
        "tensor_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "cfg_parallel_size",
        "request_type_id",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "repeat_count",
        "latency_seconds_avg",
        "latency_seconds_min",
        "latency_seconds_max",
        "latency_ms_avg",
        "latency_ms_min",
        "latency_ms_max",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote aggregated CSV: {output_csv}")


if __name__ == "__main__":
    main()
