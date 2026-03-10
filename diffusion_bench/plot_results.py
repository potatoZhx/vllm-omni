#!/usr/bin/env python3
"""
根据 runs.json 画图：横轴固定为 RPS，纵轴为指定指标。支持单图或多图（按类型拆成 latency / throughput / slo）。
可独立使用，不依赖 run_bench.sh。详见 plot_readme.md。

输出命名：单图 <prefix>.<format> 或 --output 完整路径；多图 <prefix>_latency.<format> 等。
规则：--output 与 --output-dir/--output-prefix/--format 互斥，同时指定会报错。
数据：同一 rps 多条记录时，按 rps 分组对各指标取算术平均值后再绘图。

示例：
  python3 plot_results.py -i runs.json -o figs/benchmark.png
  python3 plot_results.py -i runs.json -o figs/benchmark.png --split-by-type
  python3 plot_results.py -i runs.json --output-dir figs --output-prefix qwen --split-by-type --dpi 200
"""

import argparse
import json
import sys
from pathlib import Path

# 唯一别名表：用户/CLI 名称 -> runs.json 内规范字段名
METRIC_ALIASES = {
    "LatencyMean": "latency_mean",
    "latency_mean": "latency_mean",
    "latency_median": "latency_median",
    "latency_p50": "latency_p50",
    "LatencyP95": "latency_p95",
    "latency_p95": "latency_p95",
    "LatencyP99": "latency_p99",
    "latency_p99": "latency_p99",
    "ThroughputQps": "throughput_qps",
    "throughput_qps": "throughput_qps",
    "slo_attainment_rate": "slo_attainment_rate",
    "duration": "duration",
    "completed_requests": "completed_requests",
    "failed_requests": "failed_requests",
}

# 图例显示名（规范字段名 -> 友好名称）
METRIC_DISPLAY_NAMES = {
    "latency_mean": "Latency Mean",
    "latency_median": "Latency Median",
    "latency_p50": "Latency P50",
    "latency_p95": "Latency P95",
    "latency_p99": "Latency P99",
    "throughput_qps": "Throughput (QPS)",
    "slo_attainment_rate": "SLO Attainment Rate",
    "duration": "Duration",
    "completed_requests": "Completed Requests",
    "failed_requests": "Failed Requests",
}

DEFAULT_METRICS = ["LatencyP95"]

# 多图分组：仅 latency / throughput / slo
METRIC_GROUPS = {
    "latency": ["latency_mean", "latency_median", "latency_p50", "latency_p95", "latency_p99"],
    "throughput": ["throughput_qps"],
    "slo": ["slo_attainment_rate"],
}

DEFAULT_METRICS_BY_GROUP = {
    "latency": ["LatencyMean", "LatencyP95", "LatencyP99"],
    "throughput": ["ThroughputQps"],
    "slo": ["slo_attainment_rate"],
}

# 规范字段名 -> run 中可能出现的键名列表（便于兼容仅含 camelCase 的 JSON）
CANONICAL_TO_ALIASES: dict[str, list[str]] = {}
for alias, canonical in METRIC_ALIASES.items():
    CANONICAL_TO_ALIASES.setdefault(canonical, []).append(alias)


def _parse_metrics(s: str | list | None) -> list[str]:
    """解析为指标列表；支持 "A,B,C" 或 ["A","B","C"]，返回去重列表。"""
    if s is None or (isinstance(s, list) and len(s) == 0):
        return list(DEFAULT_METRICS)
    if isinstance(s, list):
        out = []
        for x in s:
            out.extend(y.strip() for y in str(x).split(",") if y.strip())
    else:
        out = [x.strip() for x in str(s).split(",") if x.strip()]
    seen = set()
    unique = []
    for m in out:
        if m and m not in seen:
            seen.add(m)
            unique.append(m)
    return unique if unique else list(DEFAULT_METRICS)


def _canonical(key: str) -> str | None:
    """用户指标名 -> 规范字段名；未知则返回 None。"""
    return METRIC_ALIASES.get(key)


def _get_val(run: dict, canonical_key: str):
    """从单条 run 中取规范字段值；兼容 run 内为 camelCase 或 snake_case。"""
    v = run.get(canonical_key)
    if v is not None:
        return v
    for alias in CANONICAL_TO_ALIASES.get(canonical_key, []):
        if alias != canonical_key and run.get(alias) is not None:
            return run.get(alias)
    return None


def _display_name(canonical_key: str) -> str:
    """规范字段名 -> 图例显示名。"""
    return METRIC_DISPLAY_NAMES.get(canonical_key, canonical_key)


def _metric_group(canonical_key: str) -> str | None:
    """规范字段名 -> 所属分组（latency/throughput/slo）。"""
    for group_name, keys in METRIC_GROUPS.items():
        if canonical_key in keys:
            return group_name
    return None


def _load_and_validate_runs(input_path: Path) -> list[dict]:
    """加载 runs.json，校验为非空 list[dict]，返回列表。"""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Invalid runs.json format: expected a non-empty list of objects")
    for i, r in enumerate(data):
        if not isinstance(r, dict):
            raise ValueError(f"Invalid runs.json format: item {i} is not an object")
    return data


def _filter_valid_rps(runs: list[dict]) -> list[dict]:
    """只保留含有效 rps 的记录；缺失 rps 的跳过。"""
    out = []
    for r in runs:
        rps_raw = r.get("rps")
        if rps_raw is None or rps_raw == "":
            continue
        try:
            float(rps_raw)
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def _aggregate_by_rps(runs: list[dict], metric_keys: list[str]) -> list[dict]:
    """同一 rps 多条记录时，按 rps 分组并对各指标取均值，返回 [{rps, m1, m2, ...}, ...]。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        try:
            rps_f = float(r.get("rps"))
        except (TypeError, ValueError):
            continue
        groups[rps_f].append(r)
    aggregated = []
    for rps_f in sorted(groups.keys()):
        rows = groups[rps_f]
        row = {"rps": rps_f}
        for k in metric_keys:
            vals = []
            for r in rows:
                v = _get_val(r, k)
                if v is None or v == "":
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv == fv:  # not NaN
                    vals.append(fv)
            if vals:
                row[k] = sum(vals) / len(vals)
        aggregated.append(row)
    return aggregated


def _ylabel_for_canonicals(canonical_list: list[str]) -> str:
    """根据指标类型确定单图 y 轴标签。"""
    groups = {_metric_group(c) for c in canonical_list if _metric_group(c)}
    if len(groups) == 1:
        g = next(iter(groups))
        if g == "latency":
            return "Latency (s)"
        if g == "throughput":
            return "Throughput (req/s)"
        if g == "slo":
            return "SLO Attainment Rate"
    return "Metric value"


def _draw_one_figure(
    runs: list[dict],
    canonical_metrics: list[str],
    title: str,
    xlabel: str = "RPS (requests per second)",
    ylabel: str = "Metric value",
    ylim: tuple[float, float] | None = None,
):
    """画一张图（多条曲线），constrained layout，图例优先外置；返回 figure。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for ckey in canonical_metrics:
        rps_vals = []
        metric_vals = []
        for r in runs:
            rps_raw = r.get("rps")
            if rps_raw is None or rps_raw == "":
                continue
            try:
                rps_f = float(rps_raw)
            except (TypeError, ValueError):
                continue
            v = _get_val(r, ckey)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:
                continue
            rps_vals.append(rps_f)
            metric_vals.append(fv)
        if rps_vals:
            ax.plot(rps_vals, metric_vals, marker="o", label=_display_name(ckey))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    return fig


def _group_has_data(runs: list[dict], canonical_list: list[str]) -> bool:
    """该组是否有任意指标在 runs 中存在有效数据。"""
    for ckey in canonical_list:
        for r in runs:
            if r.get("rps") is None or r.get("rps") == "":
                continue
            v = _get_val(r, ckey)
            if v is not None and v != "":
                try:
                    fv = float(v)
                    if fv == fv:
                        return True
                except (TypeError, ValueError):
                    pass
    return False


def plot_results(
    input_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path = ".",
    output_prefix: str = "benchmark",
    fmt: str = "png",
    dpi: int = 150,
    metrics: str | list[str] | None = None,
    title: str = "Diffusion serving benchmark",
    split_by_type: bool = False,
) -> str | list[str] | None:
    """
    根据 runs.json 画图并保存或显示。

    参数:
        input_path: 输入 runs.json 路径
        output: 完整输出文件路径；与 output_dir/output_prefix/fmt 互斥，不可同时指定
        output_dir: 输出目录，默认 "."
        output_prefix: 输出文件名前缀，默认 "benchmark"
        fmt: 图片格式，默认 "png"
        dpi: 保存图片 DPI，默认 150
        metrics: 纵轴指标；多图且未指定时使用各组默认
        title: 图标题；多图时自动追加 "- Latency" 等
        split_by_type: 是否按类型拆成多图（latency / throughput / slo）

    数据语义：当 runs.json 中存在相同 rps 的多条记录时，会按 rps 分组，对每个指标取算术平均值后再绘图。

    返回:
        单图保存: 返回 str（路径）
        多图保存: 返回 list[str]
        仅显示不保存: 返回 None
    """
    input_path = Path(input_path)
    runs = _load_and_validate_runs(input_path)
    runs = _filter_valid_rps(runs)
    if not runs:
        raise ValueError("No valid records with 'rps' found in runs.json")
    runs = sorted(runs, key=lambda r: float(r.get("rps")))

    try:
        import matplotlib
        if output or split_by_type:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib not installed. pip install matplotlib") from e

    # 解析输出路径：--output 优先
    if output:
        out_path = Path(output)
        out_dir = out_path.parent
        out_stem = out_path.stem
        out_fmt = out_path.suffix.lstrip(".") or fmt
    else:
        out_dir = Path(output_dir)
        out_stem = output_prefix
        out_fmt = fmt or "png"

    def single_path():
        return out_dir / f"{out_stem}.{out_fmt}"

    def multi_path(group: str):
        return out_dir / f"{out_stem}_{group}.{out_fmt}"

    if split_by_type:
        if metrics is None or (isinstance(metrics, list) and len(metrics) == 0):
            group_to_canonical = {
                "latency": [c for m in DEFAULT_METRICS_BY_GROUP["latency"] for c in [_canonical(m)] if c],
                "throughput": [c for m in DEFAULT_METRICS_BY_GROUP["throughput"] for c in [_canonical(m)] if c],
                "slo": [c for m in DEFAULT_METRICS_BY_GROUP["slo"] for c in [_canonical(m)] if c],
            }
        else:
            requested = _parse_metrics(metrics)
            group_to_canonical = {"latency": [], "throughput": [], "slo": []}
            for m in requested:
                ckey = _canonical(m)
                if ckey is None:
                    continue
                g = _metric_group(ckey)
                if g is not None and ckey not in group_to_canonical[g]:
                    group_to_canonical[g].append(ckey)
        saved = []
        for group_name in ("latency", "throughput", "slo"):
            canonical_list = group_to_canonical.get(group_name, [])
            canonical_list = [c for c in canonical_list if c is not None]
            if not canonical_list:
                continue
            if not _group_has_data(runs, canonical_list):
                print(f"Skip group '{group_name}': no valid data found", file=sys.stderr)
                continue
            agg = _aggregate_by_rps(runs, canonical_list)
            if not agg:
                print(f"Skip group '{group_name}': no valid data after aggregation", file=sys.stderr)
                continue
            ylabel = "Latency (s)" if group_name == "latency" else ("Throughput (req/s)" if group_name == "throughput" else "SLO Attainment Rate")
            title_suffix = "SLO" if group_name == "slo" else group_name.capitalize()
            ylim = (0.0, 1.1) if group_name == "slo" else None
            fig = _draw_one_figure(agg, canonical_list, f"{title} - {title_suffix}", ylabel=ylabel, ylim=ylim)
            path = multi_path(group_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(str(path))
        if not saved and not output:
            return None
        return saved if saved else None

    # 单图
    user_metrics = _parse_metrics(metrics)
    canonical_list = []
    for m in user_metrics:
        c = _canonical(m)
        if c is not None:
            canonical_list.append(c)
    if not canonical_list:
        raise ValueError("No valid metrics requested; use --list-metrics to see available keys")
    if not _group_has_data(runs, canonical_list):
        raise ValueError(f"No valid data found for requested metrics: {', '.join(user_metrics)}")
    agg = _aggregate_by_rps(runs, canonical_list)
    if not agg:
        raise ValueError(f"No valid data found for requested metrics: {', '.join(user_metrics)}")
    ylabel = _ylabel_for_canonicals(canonical_list)
    ylim = (0.0, 1.1) if set(canonical_list) <= {"slo_attainment_rate"} else None
    fig = _draw_one_figure(agg, canonical_list, title, ylabel=ylabel, ylim=ylim)
    if output is not None:
        path = single_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return str(path)
    plt.show()
    plt.close(fig)
    return None


def list_metrics(input_path: str | Path) -> list[str]:
    """列出 runs.json 中所有记录出现的 key 的并集（保证对外展示的指标列表完整）。"""
    input_path = Path(input_path)
    runs = _load_and_validate_runs(input_path)
    keys = set()
    for r in runs:
        if isinstance(r, dict):
            keys.update(r.keys())
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(
        description="Plot diffusion benchmark runs.json: x-axis=RPS, y-axis=metrics. Single or split by type (latency/throughput/slo)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to runs.json")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Full output file path. Single: use as-is; multi: stem used as prefix (e.g. figs/qwen.png -> figs/qwen_latency.png, ...)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: .). Ignored if --output is set.")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix (default: benchmark). Ignored if --output is set.")
    parser.add_argument("--format", default=None, help="Image format: png, pdf, svg, etc. (default: png). Cannot combine with --output.")
    parser.add_argument("--dpi", type=int, default=None, help="DPI for saved image (default: 150).")
    parser.add_argument(
        "--metrics", "-m",
        nargs="*",
        default=None,
        help="Y-axis metrics. Default: LatencyP95. E.g. -m LatencyP95 or -m 'LatencyP95,ThroughputQps'",
    )
    parser.add_argument("--title", type=str, default="Diffusion serving benchmark", help="Plot title")
    parser.add_argument(
        "--split-by-type",
        action="store_true",
        help="Split into multiple figures: <prefix>_latency.<format>, <prefix>_throughput.<format>, <prefix>_slo.<format>",
    )
    parser.add_argument("--list-metrics", action="store_true", help="List available metric keys and exit")
    args = parser.parse_args()

    if args.list_metrics:
        keys = list_metrics(args.input)
        print("Available keys in runs.json (for --metrics):")
        for k in keys:
            print(" ", k)
        return

    # --output 与 --output-dir/--output-prefix/--format 互斥，避免悄悄忽略
    if args.output is not None and any([
        args.output_dir is not None,
        args.output_prefix is not None,
        args.format is not None,
    ]):
        print("Error: --output cannot be used together with --output-dir, --output-prefix, or --format", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir is not None else "."
    output_prefix = args.output_prefix if args.output_prefix is not None else "benchmark"
    fmt = args.format if args.format is not None else "png"
    dpi = args.dpi if args.dpi is not None else 150

    try:
        result = plot_results(
            args.input,
            output=args.output,
            output_dir=output_dir,
            output_prefix=output_prefix,
            fmt=fmt,
            dpi=dpi,
            metrics=args.metrics,
            title=args.title,
            split_by_type=args.split_by_type,
        )
        if result is not None:
            if isinstance(result, list):
                print("Saved:", ", ".join(result))
            else:
                print("Saved", result)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
