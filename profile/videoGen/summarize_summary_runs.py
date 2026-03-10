#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read summary_runs.csv, compute fastest parallel config per request type, "
            "and output a request_type x parallel_config latency table."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="profile/videoGen/results/20260308_121123/summary_runs.csv",
        help="Path to summary_runs.csv.",
    )
    parser.add_argument(
        "--pivot-output-csv",
        default="profile/videoGen/results/20260308_121123/latency_pivot_min_seconds.csv",
        help="Output CSV for latency pivot table (rows=request_type_id, cols=parallel_name).",
    )
    parser.add_argument(
        "--fastest-output-csv",
        default="profile/videoGen/results/20260308_121123/fastest_config_by_request_type.csv",
        help="Output CSV for fastest config by request_type_id.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision for output values (default: 4).",
    )
    return parser.parse_args()


def resolve_path(workspace_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (workspace_root / path).resolve()
    return path


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    input_csv = resolve_path(workspace_root, args.input_csv)
    pivot_output = resolve_path(workspace_root, args.pivot_output_csv)
    fastest_output = resolve_path(workspace_root, args.fastest_output_csv)

    min_latency_by_req_cfg: dict[str, dict[str, float]] = defaultdict(dict)
    cfg_names: set[str] = set()

    total_rows = 0
    ok_rows = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row.get("status") != "ok":
                continue

            request_type_id = (row.get("request_type_id") or "").strip()
            parallel_name = (row.get("parallel_name") or "").strip()
            latency_seconds = to_float((row.get("latency_seconds") or "").strip())

            if not request_type_id or not parallel_name or latency_seconds is None:
                continue

            ok_rows += 1
            cfg_names.add(parallel_name)

            current = min_latency_by_req_cfg[request_type_id].get(parallel_name)
            if current is None or latency_seconds < current:
                min_latency_by_req_cfg[request_type_id][parallel_name] = latency_seconds

    request_type_ids = sorted(min_latency_by_req_cfg.keys())
    sorted_cfg_names = sorted(cfg_names)

    pivot_output.parent.mkdir(parents=True, exist_ok=True)
    with pivot_output.open("w", encoding="utf-8", newline="") as f:
        fields = ["request_type_id", *sorted_cfg_names]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for req_id in request_type_ids:
            row = {"request_type_id": req_id}
            for cfg_name in sorted_cfg_names:
                value = min_latency_by_req_cfg[req_id].get(cfg_name)
                row[cfg_name] = "" if value is None else f"{value:.{args.precision}f}"
            writer.writerow(row)

    fastest_output.parent.mkdir(parents=True, exist_ok=True)
    with fastest_output.open("w", encoding="utf-8", newline="") as f:
        fields = ["request_type_id", "fastest_parallel_name", "min_latency_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for req_id in request_type_ids:
            items = list(min_latency_by_req_cfg[req_id].items())
            if not items:
                continue
            fastest_cfg, min_latency = min(items, key=lambda x: x[1])
            writer.writerow(
                {
                    "request_type_id": req_id,
                    "fastest_parallel_name": fastest_cfg,
                    "min_latency_seconds": f"{min_latency:.{args.precision}f}",
                }
            )

    print(f"Input CSV: {input_csv}")
    print(f"Total rows: {total_rows}, valid ok rows: {ok_rows}")
    print(f"Request types: {len(request_type_ids)}, parallel configs: {len(sorted_cfg_names)}")
    print(f"Wrote pivot CSV: {pivot_output}")
    print(f"Wrote fastest CSV: {fastest_output}")


if __name__ == "__main__":
    main()
