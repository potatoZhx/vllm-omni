import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def detect_task_type(model: str) -> str:
    """
    Map model path/name to task_type.

    Rule:
    - if "wan" (case-insensitive) appears in model string -> "video"
    - otherwise -> "image"
    """
    if "wan" in model.lower():
        return "video"
    return "image"


def _require_nonempty(row: Dict[str, str], name: str, row_idx: int) -> str:
    """Fail fast: raise if field is missing or empty."""
    value = row.get(name, "")
    if value == "" or value is None:
        raise ValueError(
            f"Row {row_idx + 1}: required field '{name}' is missing or empty. Got: {repr(row)}"
        )
    return value


def _require_int(row: Dict[str, str], name: str, row_idx: int) -> int:
    """Fail fast: raise if field is missing, empty, or not convertible to int."""
    value = _require_nonempty(row, name, row_idx)
    try:
        return int(float(value))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Row {row_idx + 1}: field '{name}' must be numeric, got {repr(value)}") from e


def _require_latency_s(row: Dict[str, str], row_idx: int) -> float:
    """Fail fast: require latency_seconds or latency_ms, return latency_s (秒)."""
    lat_sec = row.get("latency_seconds", "")
    lat_ms = row.get("latency_ms", "")
    if lat_sec != "" and lat_sec is not None:
        try:
            return float(lat_sec)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Row {row_idx + 1}: latency_seconds must be numeric, got {repr(lat_sec)}"
            ) from e
    if lat_ms != "" and lat_ms is not None:
        try:
            return float(lat_ms) / 1000.0
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Row {row_idx + 1}: latency_ms must be numeric, got {repr(lat_ms)}"
            ) from e
    raise ValueError(
        f"Row {row_idx + 1}: require 'latency_seconds' or 'latency_ms'. Got: {repr(row)}"
    )


def row_to_profile(row: Dict[str, str], row_idx: int = 0) -> Dict[str, Any]:
    """
    转换为 newest_profile_A100.json 格式，与 simulation 查表对齐。
    输出 latency_s（秒），非 latency_ms。
    """
    parallel_name = _require_nonempty(row, "parallel_name", row_idx)
    model = _require_nonempty(row, "model", row_idx)

    profile: Dict[str, Any] = {
        "instance_type": parallel_name,
        "task_type": detect_task_type(model),
        "width": _require_int(row, "width", row_idx),
        "height": _require_int(row, "height", row_idx),
        "num_frames": _require_int(row, "num_frames", row_idx),
        "steps": _require_int(row, "num_inference_steps", row_idx),
        "latency_s": _require_latency_s(row, row_idx),
    }
    return profile


def convert_csv_to_json(input_csv: Path, output_json: Path) -> None:
    profiles: List[Dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            profiles.append(row_to_profile(row, row_idx=i))

    data = {"profiles": profiles}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert summary CSV to profiles JSON compatible with instance_runtime_profile.sample.json."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file. "
            "If omitted, defaults to 'csv_to_profiles.yaml' in the same directory as this script."
        ),
    )
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to use --config. Please install it with `pip install pyyaml`.") from exc

    if args.config is None:
        # 默认使用脚本同目录下的 csv_to_profiles.yaml
        cfg_path = Path(__file__).with_name("csv_to_profiles.yaml")
    else:
        cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_path = cfg.get("input_csv")
    output_path = cfg.get("output_json")

    if not input_path or not output_path:
        raise SystemExit("Config YAML must contain 'input_csv' and 'output_json' fields.")

    input_csv = (cfg_path.parent / input_path).resolve() if not Path(input_path).is_absolute() else Path(input_path)
    output_json = (cfg_path.parent / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)

    convert_csv_to_json(input_csv, output_json)


if __name__ == "__main__":
    main()

