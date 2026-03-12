#!/usr/bin/env python3
"""
离散事件模拟器：单 YAML 配置 + 单脚本，与 README 设计一致。
用法：python simulation/simulation.py [simulation_config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# 可选：无 PyYAML 时提示
try:
    import yaml
except ImportError:
    yaml = None

INF = float("inf")
PROFILE_STEPS = (1, 5, 10, 30, 50)


def bucket_steps(steps: int) -> int:
    """将请求的 steps 映射到 profile 档位。查不到则报错。"""
    if steps <= 1:
        return 1
    if 2 <= steps <= 6:
        return 5
    if 7 <= steps <= 24:
        return 10
    if 25 <= steps <= 35:
        return 30
    if 36 <= steps <= 50:
        return 50
    raise ValueError(f"steps={steps} 超出支持的档位范围，请使用 1 或 4-50 以内")


def worker_config_id(w: dict) -> str:
    """Worker 三整数 sp, cfg, tp 拼接为 config_id。"""
    return f"sp{w['sp']}_cfg{w['cfg']}_tp{w['tp']}"


def load_profile(profile_path: Path) -> dict[tuple[str, int, str], float]:
    """加载 profile CSV，键 (size, steps, config_id) -> request_time_s。查表严格，无则报错。"""
    table = {}
    with open(profile_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            size = row["size"].strip()
            steps = int(row["steps"])
            config_id = row["config_id"].strip()
            request_time_s = float(row["request_time_s"])
            table[(size, steps, config_id)] = request_time_s
    return table


def lookup(
    table: dict[tuple[str, int, str], float],
    size: str,
    steps: int,
    config_id: str,
) -> float:
    """查表得到 request_time_s。size 或 config 无匹配则报错。"""
    bucketed = bucket_steps(steps)
    key = (size, bucketed, config_id)
    if key not in table:
        raise KeyError(f"profile 中无匹配: size={size!r}, steps(档位)={bucketed}, config_id={config_id!r}")
    return table[key]


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置。"""
    if yaml is None:
        raise RuntimeError("请安装 PyYAML: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    sim = cfg.get("simulation", {})
    profile_path = sim.get("profile_path", "profile_qwen.csv")
    if not Path(profile_path).is_absolute():
        profile_path = base / profile_path
    sim["profile_path"] = Path(profile_path)
    if sim.get("output_dir"):
        out = Path(sim["output_dir"])
        if not out.is_absolute():
            out = base / out
        sim["output_dir"] = out
    trace_path = sim.get("trace_path")
    if trace_path is not None:
        p = Path(trace_path)
        if not p.is_absolute():
            p = base / trace_path
        sim["trace_path"] = p
    return cfg


def _parse_sd3_trace_line(line: str) -> dict | None:
    """解析 trace 单行，只取 size、steps（顺序）；timestamp 不使用。"""
    if not line.strip().startswith("Request("):
        return None
    h = re.search(r"height=(\d+)", line)
    w = re.search(r"width=(\d+)", line)
    s = re.search(r"num_inference_steps=(\d+)", line)
    if not all([h, w, s]):
        return None
    height, width = int(h.group(1)), int(w.group(1))
    return {"size": f"{height}x{width}", "steps": int(s.group(1))}


def load_trace_template(trace_path: Path) -> list[dict]:
    """从 trace 按文件顺序加载 (size, steps) 列表；仅用于请求类型顺序，不使用 timestamp。"""
    template = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            r = _parse_sd3_trace_line(line)
            if r is not None:
                template.append(r)
    return template


def build_requests(cfg: dict, rps: float, t_end: float) -> list[dict]:
    """生成请求列表。时间驱动：t=0, 1/rps, 2/rps, ... 当 t <= T_end 时生成该请求（与 README 一致）。trace 仅提供请求顺序对应的 size、steps。"""
    sim = cfg["simulation"]
    default = cfg.get("default_request", {})
    size_default = default.get("size", "128x128")
    steps_default = int(default.get("steps", 5))
    trace_path = sim.get("trace_path")
    if trace_path and Path(trace_path).exists():
        template = load_trace_template(trace_path)
        if not template:
            raise ValueError(f"trace 解析后无有效请求: {trace_path}")
    else:
        template = [{"size": size_default, "steps": steps_default}]
    requests = []
    i = 0
    t = 0.0
    while t <= t_end and rps > 0:
        req = template[i % len(template)].copy()
        req["request_id"] = f"req_{i}"
        req["arrival_time"] = t
        requests.append(req)
        i += 1
        t = i / rps
    if not requests and rps > 0:
        req = template[0].copy()
        req["request_id"] = "req_0"
        req["arrival_time"] = 0.0
        requests.append(req)
    return requests


def assign_slo(requests: list[dict], table: dict, workers: list[dict], slo_scale: float) -> None:
    """为每个请求赋 slo_ms：用第一个 worker 的 config 查表 * 1000 * slo_scale。"""
    if not workers or slo_scale <= 0:
        return
    cid = worker_config_id(workers[0])
    for r in requests:
        bt = bucket_steps(r["steps"])
        key = (r["size"], bt, cid)
        if key not in table:
            raise KeyError(f"为 SLO 查表失败: size={r['size']!r}, steps(档位)={bt}, config_id={cid!r}")
        r["slo_ms"] = table[key] * 1000.0 * slo_scale


# ---------- 调度模块（可插拔） ----------

def dispatch_round_robin(request: dict, workers: list, state: dict) -> int:
    """轮询：若有空闲 worker 则仅在空闲 worker 中轮询；若全部忙碌则在全体 worker 中轮询。"""
    n = len(workers)
    available = [i for i in range(n) if workers[i]["next_time"] == INF]
    cursor = state.setdefault("next_index", 0)
    if available:
        selected = available[cursor % len(available)]
        state["next_index"] = cursor + 1
        return selected
    selected = cursor % n
    state["next_index"] = cursor + 1
    return selected


def dispatch_fcfs(request: dict, workers: list, state: dict) -> int:
    """先到先服务：优先选配置顺序里第一个空闲的 worker；若全部忙碌则退化为当前负载最小的 worker。"""
    for i, w in enumerate(workers):
        if w["next_time"] == INF:
            return i
    best = 0
    best_load = INF
    for i, w in enumerate(workers):
        load = len(w["queue"]) + (0 if w["next_time"] == INF else 1)
        if load < best_load:
            best_load = load
            best = i
    return best


def _remaining_work_s(worker: dict, table: dict, current_time: float) -> float:
    """该 worker 当前剩余总工作量（秒）：正在执行剩余时间 + 队列中所有请求的 service_time 之和。"""
    running = 0.0
    if worker["next_time"] != INF:
        running = max(0.0, worker["next_time"] - current_time)
    queued = 0.0
    for r in worker["queue"]:
        try:
            queued += lookup(table, r["size"], r["steps"], worker["config_id"])
        except KeyError:
            pass
    return running + queued


def dispatch_short_queue_runtime(request: dict, workers: list, state: dict) -> int:
    """最短队列预估时间：选当前剩余总工作量（秒）最小的实例。基于真实队列内请求类型查表，非 queue_len×本请求时间。"""
    table = state["table"]
    t = state.get("current_time", 0.0)
    best = 0
    best_work = INF
    for i, w in enumerate(workers):
        work = _remaining_work_s(w, table, t)
        if work < best_work:
            best_work = work
            best = i
    return best


def dispatch_estimated_completion_time(request: dict, workers: list, state: dict) -> int:
    """预估完成时间最小：选（当前剩余总工作量 + 本请求在该实例的 service_time）最小的实例。"""
    table = state["table"]
    size, steps = request["size"], request["steps"]
    t = state.get("current_time", 0.0)
    best = 0
    best_ect = INF
    for i, w in enumerate(workers):
        remaining = _remaining_work_s(w, table, t)
        try:
            my_st = lookup(table, size, steps, w["config_id"])
        except KeyError:
            my_st = INF
        ect = remaining + my_st
        if ect < best_ect:
            best_ect = ect
            best = i
    return best


DISPATCH = {
    "round_robin": dispatch_round_robin,
    "fcfs": dispatch_fcfs,
    "short_queue_runtime": dispatch_short_queue_runtime,
    "estimated_completion_time": dispatch_estimated_completion_time,
}

def run_simulation(
    requests: list[dict],
    workers: list[dict],
    table: dict[tuple[str, int, str], float],
    rps: float,
    t_end: float,
    algorithm: str,
) -> dict:
    """单次模拟，返回与 plot_results 对齐的指标 + algorithm。

    入队后、立即派工前由调度器对队列排序的扩展点为占位：当前默认行为仅为将任务放到队列末尾，以后可在此处扩展。
    """
    if algorithm not in DISPATCH:
        raise ValueError(f"不支持的算法: {algorithm}，可选: {list(DISPATCH.keys())}")
    dispatch_fn = DISPATCH[algorithm]
    state = {"table": table}

    # 复制 worker 状态：config_id, queue, next_time
    ws = []
    for w in workers:
        ws.append({
            "config_id": worker_config_id(w),
            "queue": [],
            "next_time": INF,
        })
    n_workers = len(ws)
    # 调度器下一事件 = 下一请求的到达时间（请求发起时间），不是 rps 均匀间隔
    req_index = [0]

    def scheduler_next_time():
        if req_index[0] >= len(requests):
            return INF
        t = requests[req_index[0]]["arrival_time"]
        return t if t <= t_end else INF

    scheduler_next = scheduler_next_time()
    completed = []

    def next_request():
        if req_index[0] >= len(requests):
            return None
        r = requests[req_index[0]].copy()
        req_index[0] += 1
        return r

    for w in ws:
        w["current_request"] = None

    while True:
        t = min(scheduler_next, min((w["next_time"] for w in ws)))
        if t == INF and all(w["next_time"] == INF and len(w["queue"]) == 0 for w in ws):
            break

        # 请求到达：在 arrival_time 将请求入队（请求发起时间，此时不一定被执行）
        if t == scheduler_next and req_index[0] < len(requests):
            req = next_request()
            if req is not None:
                state["current_time"] = t
                wi = dispatch_fn(req, ws, state)
                req["assigned_worker_index"] = wi
                req["assigned_worker_config"] = ws[wi]["config_id"]
                # 默认行为：任务放到队列末尾。入队后、立即派工前由调度器对队列排序为占位，以后可在此扩展。
                ws[wi]["queue"].append(req)
            scheduler_next = scheduler_next_time()

        # Worker 完成：request_time_s 为从开始执行到执行结束的时间，finish_time = 当前时钟
        for w in ws:
            if w["next_time"] == t:
                rec = w.get("current_request")
                w["next_time"] = INF
                w["current_request"] = None
                if rec is not None:
                    rec["finish_time"] = t
                    rec["latency"] = t - rec["arrival_time"]
                    completed.append(rec)

        # 立即派工：空闲 worker 从队列取请求，此时才记录 start_time（开始执行时间）
        for w in ws:
            if w["next_time"] == INF and w["queue"]:
                req = w["queue"].pop(0)
                req["start_time"] = t  # 从本时刻开始执行
                st = lookup(table, req["size"], req["steps"], w["config_id"])
                w["next_time"] = t + st
                w["current_request"] = req

    # 未完成的请求：仍在队列中的不记入 completed，无 failed 计数（按 README 可扩展）
    n_ok = len(completed)
    for r in completed:
        if "latency" not in r and "finish_time" in r and "arrival_time" in r:
            r["latency"] = r["finish_time"] - r["arrival_time"]
        if "start_time" in r and "arrival_time" in r:
            r["waiting_time"] = r["start_time"] - r["arrival_time"]
        if "finish_time" in r and "start_time" in r:
            r["service_time"] = r["finish_time"] - r["start_time"]

    # 每个请求的明细：分配给的 worker、发起时间、开始执行时间、完成时间等
    def _round4(x):
        return round(x, 4) if isinstance(x, (int, float)) else x

    per_request = []
    for r in completed:
        per_request.append({
            "request_id": r.get("request_id"),
            "assigned_worker_index": r.get("assigned_worker_index"),
            "assigned_worker_config": r.get("assigned_worker_config"),
            "arrival_time": _round4(r["arrival_time"]),
            "start_time": _round4(r["start_time"]),
            "finish_time": _round4(r["finish_time"]),
            "latency": _round4(r["latency"]),
            "waiting_time": _round4(r["waiting_time"]),
            "service_time": _round4(r["service_time"]),
            "size": r.get("size"),
            "steps": r.get("steps"),
        })
    per_request.sort(key=lambda x: (x["arrival_time"], x.get("request_id") or ""))

    latencies = [r["latency"] for r in completed if "latency" in r]
    waiting_times = [r["waiting_time"] for r in completed if "waiting_time" in r]
    service_times = [r["service_time"] for r in completed if "service_time" in r]

    def perc(sort_list: list, p: float) -> float:
        """分位数，与 numpy.percentile 默认线性插值一致。"""
        if not sort_list:
            return 0.0
        return float(np.percentile(sort_list, p * 100.0))

    if n_ok == 0:
        duration = throughput_qps = 0.0
        latency_mean = latency_median = latency_p50 = latency_p95 = latency_p99 = 0.0
        waiting_time_mean = waiting_time_p95 = waiting_time_p99 = 0.0
        service_time_mean = service_time_p95 = service_time_p99 = 0.0
        slo_attainment_rate = 0.0
    else:
        t_first = min(r["arrival_time"] for r in completed)
        t_last = max(r["finish_time"] for r in completed)
        duration = t_last - t_first
        throughput_qps = n_ok / duration if duration > 0 else 0.0
        latencies.sort()
        latency_mean = sum(latencies) / n_ok
        latency_median = float(np.median(latencies))
        latency_p50 = perc(latencies, 0.50)
        latency_p95 = perc(latencies, 0.95)
        latency_p99 = perc(latencies, 0.99)
        waiting_times.sort()
        waiting_time_mean = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
        waiting_time_p95 = perc(waiting_times, 0.95)
        waiting_time_p99 = perc(waiting_times, 0.99)
        service_times.sort()
        service_time_mean = sum(service_times) / len(service_times) if service_times else 0.0
        service_time_p95 = perc(service_times, 0.95)
        service_time_p99 = perc(service_times, 0.99)
        slo_met = sum(1 for r in completed if r.get("slo_ms") is not None and r["latency"] * 1000 <= r["slo_ms"])
        slo_total = sum(1 for r in completed if r.get("slo_ms") is not None)
        slo_attainment_rate = (slo_met / slo_total) if slo_total else 0.0

    return {
        "algorithm": algorithm,
        "rps": rps,
        "duration": round(duration, 4),
        "completed_requests": n_ok,
        "failed_requests": 0,
        "throughput_qps": round(throughput_qps, 4),
        "latency_mean": round(latency_mean, 4),
        "latency_median": round(latency_median, 4),
        "latency_p50": round(latency_p50, 4),
        "latency_p95": round(latency_p95, 4),
        "latency_p99": round(latency_p99, 4),
        "waiting_time_mean": round(waiting_time_mean, 4),
        "waiting_time_p95": round(waiting_time_p95, 4),
        "waiting_time_p99": round(waiting_time_p99, 4),
        "service_time_mean": round(service_time_mean, 4),
        "service_time_p95": round(service_time_p95, 4),
        "service_time_p99": round(service_time_p99, 4),
        "slo_attainment_rate": round(slo_attainment_rate, 4),
        "requests": per_request,
    }


REQUEST_CSV_COLUMNS = (
    "algorithm",
    "rps",
    "request_id",
    "assigned_worker_index",
    "assigned_worker_config",
    "arrival_time",
    "start_time",
    "finish_time",
    "latency",
    "waiting_time",
    "service_time",
    "size",
    "steps",
)


def _write_requests_csv(path: Path, runs: list[dict]) -> None:
    """将各 run 的 requests 明细写入 CSV，每行一个请求，含 algorithm、rps 及请求字段。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for run in runs:
            algo, rps_val = run["algorithm"], run["rps"]
            for req in run.get("requests", []):
                row = {k: req.get(k, "") for k in REQUEST_CSV_COLUMNS}
                row["algorithm"] = algo
                row["rps"] = rps_val
                w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="离散事件模拟器：YAML 配置 + profile 查表，输出与 plot_results 对齐")
    parser.add_argument("config", nargs="?", default="simulation_config.yaml", help="YAML 配置文件路径")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    sim = cfg["simulation"]
    workers = cfg["workers"]
    profile_path = sim["profile_path"]
    if not profile_path.exists():
        print(f"profile 不存在: {profile_path}", file=sys.stderr)
        sys.exit(1)

    table = load_profile(profile_path)
    t_end = float(sim["t_end"])
    rps_cfg = sim["rps"]
    rps_list = [float(x) for x in (rps_cfg if isinstance(rps_cfg, list) else [rps_cfg])]
    slo_scale = float(sim.get("slo_scale", 3))
    algorithms_to_run = cfg.get("scheduler", {}).get("algorithms_to_run", [cfg.get("scheduler", {}).get("algorithm", "round_robin")])
    if isinstance(algorithms_to_run, str):
        algorithms_to_run = [algorithms_to_run]
    output_dir = Path(sim["output_dir"])
    output_merged = sim.get("output_merged", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for algo in algorithms_to_run:
        runs_for_algo = []
        for rps in rps_list:
            requests = build_requests(cfg, rps, t_end)
            assign_slo(requests, table, workers, slo_scale)
            try:
                out = run_simulation(requests, workers, table, rps, t_end, algo)
                runs_for_algo.append(out)
                all_runs.append(out)
            except (KeyError, ValueError) as e:
                print(f"算法 {algo} rps={rps} 运行失败: {e}", file=sys.stderr)
                sys.exit(1)
        if not output_merged:
            runs_stats = [{k: v for k, v in run.items() if k != "requests"} for run in runs_for_algo]
            out_path = output_dir / f"{algo}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(runs_stats, f, indent=2)
            req_path = output_dir / f"{algo}_requests.csv"
            _write_requests_csv(req_path, runs_for_algo)
            print(f"已写入 {algo}: 统计 {out_path}，请求明细 {req_path} ({len(runs_for_algo)} 个 rps 点)")

    if output_merged and all_runs:
        merged_stats = [{k: v for k, v in run.items() if k != "requests"} for run in all_runs]
        merged_path = output_dir / "merged.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_stats, f, indent=2)
        merged_req_path = output_dir / "merged_requests.csv"
        _write_requests_csv(merged_req_path, all_runs)
        print(f"已合并写入: 统计 {merged_path}，请求明细 {merged_req_path}")

    # 跑完后自动画图（可配置）
    plot_cfg = cfg.get("plot", {})
    if plot_cfg.get("after_run", False) and all_runs:
        plot_metrics = plot_cfg.get("metrics", ["latency_p95", "latency_mean"])
        if isinstance(plot_metrics, str):
            plot_metrics = [plot_metrics]
        plot_output_dir = plot_cfg.get("output_dir", "output")
        plot_output_prefix = plot_cfg.get("output_prefix", "compare")
        base = config_path.parent
        if not Path(plot_output_dir).is_absolute():
            plot_output_dir = base / plot_output_dir
        else:
            plot_output_dir = Path(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from diffusion_bench.plot_results import plot_results as plot_results_fn
            merged_for_plot = output_dir / "_merged_for_plot.json"
            plot_runs = [{k: v for k, v in run.items() if k != "requests"} for run in all_runs]
            with open(merged_for_plot, "w", encoding="utf-8") as f:
                json.dump(plot_runs, f, indent=2)
            saved_list = []
            for metric in plot_metrics:
                out_path = plot_output_dir / f"{plot_output_prefix}_{metric}.png"
                saved = plot_results_fn(
                    merged_for_plot,
                    output=out_path,
                    metrics=[metric],
                    title="Simulation",
                    split_by_type=True,
                    group_by="algorithm",
                )
                if saved:
                    saved_list.append(saved)
            try:
                merged_for_plot.unlink()
            except OSError:
                pass
            if saved_list:
                print("已画图保存（一指标一图）:", saved_list)
        except ImportError as e:
            print("跳过自动画图（请安装 matplotlib 并在项目根目录运行）:", e, file=sys.stderr)
        except Exception as e:
            print("自动画图失败:", e, file=sys.stderr)

    print("完成。画图示例: python diffusion_bench/plot_results.py --input-dir", output_dir, "--algorithms", " ".join(algorithms_to_run), "-o figs/compare --split-by-type")


if __name__ == "__main__":
    main()
