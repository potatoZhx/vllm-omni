#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pivot CSV from one profiling results directory. "
            "Rows are request types; columns are parallel configs."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to one results directory, e.g. profile/videoGen/results/g2_1_20260305_190229",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path, e.g. profile/videoGen/video_parallel_pivot_g2.csv",
    )
    parser.add_argument(
        "--request-types-config",
        default="profile/videoGen/request_types_24.json",
        help="Request types config used to emit full row set (default: profile/videoGen/request_types_24.json).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision for stats string (default: 4).",
    )
    return parser.parse_args()


def load_request_types(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for (height, width), num_frames, num_steps in product(
        payload["height_width"],
        payload["num_frames"],
        payload["num_inference_steps"],
    ):
        rows.append(
            {
                "request_type_id": f"h{int(height)}_w{int(width)}_f{int(num_frames)}_s{int(num_steps)}",
                "height": int(height),
                "width": int(width),
                "num_frames": int(num_frames),
                "num_inference_steps": int(num_steps),
            }
        )
    rows.sort(
        key=lambda x: (
            x["height"],
            x["width"],
            x["num_frames"],
            x["num_inference_steps"],
        )
    )
    return rows


def format_stats(values: list[float], precision: int) -> str:
    if not values:
        return ""
    mean_v = statistics.mean(values)
    min_v = min(values)
    max_v = max(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return (
        f"mean={mean_v:.{precision}f}s "
        f"min={min_v:.{precision}f}s "
        f"max={max_v:.{precision}f}s "
        f"std={std_v:.{precision}f}s"
    )


def column_name_from_dir(config_dir: Path, payload: dict | None) -> str:
    if payload and isinstance(payload.get("parallel_name"), str) and payload["parallel_name"]:
        return payload["parallel_name"]
    name = config_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def main() -> None:
    args = parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (workspace_root / results_dir).resolve()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = (workspace_root / output_csv).resolve()

    req_cfg = Path(args.request_types_config)
    if not req_cfg.is_absolute():
        req_cfg = (workspace_root / req_cfg).resolve()

    request_rows = load_request_types(req_cfg)

    config_dirs = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith("g") and "_" in p.name
    )

    latencies_by_cfg_req: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    config_columns: list[str] = []

    for cfg_dir in config_dirs:
        worker_json = cfg_dir / "worker_results.json"
        payload = None
        if worker_json.exists():
            try:
                payload = json.loads(worker_json.read_text(encoding="utf-8"))
            except Exception:
                payload = None

        cfg_name = column_name_from_dir(cfg_dir, payload)
        if cfg_name not in config_columns:
            config_columns.append(cfg_name)

        if not payload:
            continue

        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            if item.get("status") != "ok":
                continue
            req_id = str(item.get("request_type_id", ""))
            if not req_id:
                continue
            try:
                latency = float(item.get("latency_seconds", ""))
            except Exception:
                continue
            latencies_by_cfg_req[cfg_name][req_id].append(latency)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "request_type_id",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        *config_columns,
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for req in request_rows:
            row = {
                "request_type_id": req["request_type_id"],
                "height": req["height"],
                "width": req["width"],
                "num_frames": req["num_frames"],
                "num_inference_steps": req["num_inference_steps"],
            }
            for cfg_name in config_columns:
                values = latencies_by_cfg_req[cfg_name].get(req["request_type_id"], [])
                row[cfg_name] = format_stats(values, args.precision)
            writer.writerow(row)

    print(f"Wrote pivot CSV: {output_csv}")
    print(f"Configs: {len(config_columns)} | Request rows: {len(request_rows)}")


if __name__ == "__main__":
    main()
