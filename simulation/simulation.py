#!/usr/bin/env python3
"""
离散事件模拟器：单 YAML 配置 + 单脚本，与 README 设计一致。
用法：python simulation/simulation.py [simulation_config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import random
from pathlib import Path

import numpy as np

# 可选：无 PyYAML 时提示
try:
    import yaml
except ImportError:
    yaml = None

INF = float("inf")


def worker_config_id(w: dict) -> str:
    """Worker 三整数 sp, cfg, tp 拼接为 config_id。"""
    return f"sp{w['sp']}_cfg{w['cfg']}_tp{w['tp']}"


def _parse_size(size: str) -> tuple[int, int]:
    """解析 size='HxW'，返回 (height, width)。"""
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", str(size))
    if not m:
        raise ValueError(f"非法 size 格式: {size!r}，期望 'HxW'")
    return int(m.group(1)), int(m.group(2))


REQUIRED_PROFILE_KEYS = ("instance_type", "task_type", "width", "height", "num_frames", "steps", "latency_s")


def load_profile_json(profile_path: Path) -> dict[tuple[str, str, int, int, int, int], float]:
    """
    加载 profile JSON，与 newest_profile_A100.json 格式严格对齐。
    键: (instance_type, task_type, width, height, num_frames, steps) -> request_time_s
    必含字段: instance_type, task_type, width, height, num_frames, steps, latency_s
    字段缺失或格式不符则 fail-fast。
    """
    with open(profile_path, encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("JSON profile 格式错误：顶层需要 'profiles' 列表")
    table = {}
    for idx, p in enumerate(profiles):
        if not isinstance(p, dict):
            raise ValueError(f"Profile 条目 {idx} 须为对象，得到 {type(p)}")
        # 严格校验：必须含 latency_s，禁止使用 latency_ms（避免单位混淆）
        if "latency_ms" in p:
            raise ValueError(
                f"Profile 条目 {idx}: 禁止 latency_ms，请使用 latency_s（秒）以对齐 newest_profile 格式"
            )
        for key in REQUIRED_PROFILE_KEYS:
            if key not in p:
                raise ValueError(
                    f"Profile 条目 {idx} 缺少必填字段 {key!r}。"
                    "需与 newest_profile_A100.json 格式对齐"
                )
        instance_type = str(p["instance_type"]).strip()
        task_type = str(p["task_type"]).strip().lower()
        width = int(p["width"])
        height = int(p["height"])
        num_frames = int(p["num_frames"])
        steps = int(p["steps"])
        try:
            latency_s = float(p["latency_s"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Profile 条目 {idx}: latency_s 须为数值，得到 {p['latency_s']!r}") from e
        key = (instance_type, task_type, width, height, num_frames, steps)
        if key in table:
            raise ValueError(f"Profile 存在重复键 {key}")
        table[key] = latency_s
    return table


def load_profile(profile_path: Path) -> dict:
    """加载 profile JSON。"""
    table = load_profile_json(profile_path)
    return {"source": "json", "json_table": table}


def lookup(profile_data: dict, request: dict, worker: dict) -> float:
    """
    按 profile 精确查表得到 request_time_s。
    字段（instance_type, width, height, num_frames, steps）必须与 profile 完全匹配，否则 fail-fast。
    task_type 由请求推断，也须在 profile 中存在对应条目。
    """
    table = profile_data["json_table"]
    instance_type = str(worker.get("instance_type") or worker.get("config_id") or "").strip()
    if not instance_type:
        raise KeyError("profile 查表失败：worker 缺少 instance_type 或 config_id")
    width = request.get("width")
    height = request.get("height")
    if width is None or height is None:
        height, width = _parse_size(request.get("size", ""))
    width, height = int(width), int(height)
    num_frames = int(request.get("num_frames", 1))
    task_type = str(request.get("task_type") or ("video" if num_frames > 1 else "image")).lower().strip()
    steps = int(request["steps"])

    key = (instance_type, task_type, width, height, num_frames, steps)
    if key not in table:
        raise KeyError(
            "profile 中无精确匹配，请检查 instance_type/width/height/num_frames/steps 与 profile 是否一致: "
            f"instance_type={instance_type!r}, task_type={task_type!r}, "
            f"width={width}, height={height}, num_frames={num_frames}, steps={steps}"
        )
    return table[key]


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置。"""
    if yaml is None:
        raise RuntimeError("请安装 PyYAML: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    sim = cfg.get("simulation", {})
    profile_path = sim.get("profile_path", "profile.json")
    if not Path(profile_path).is_absolute():
        profile_path = base / profile_path
    sim["profile_path"] = Path(profile_path)
    profile_stem = Path(profile_path).stem
    if sim.get("output_dir"):
        out = Path(sim["output_dir"])
        if not out.is_absolute():
            out = base / out
        # output_dir 为 output 根目录时，自动追加 profile 名子目录：output/{profile名}/
        if out.name == "output":
            out = out / profile_stem
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
    nf = re.search(r"num_frames=(\d+)", line)
    if not all([h, w, s]):
        return None
    height, width = int(h.group(1)), int(w.group(1))
    num_frames = int(nf.group(1)) if nf else 1
    task_type = "video" if num_frames > 1 else "image"
    return {
        "size": f"{height}x{width}",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "task_type": task_type,
        "steps": int(s.group(1)),
    }


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
    """生成请求列表。

    与 diffusion_benchmark_serving + run_global_scheduler_benchmark (fixed_duration) 对齐：
    - 请求数 N = ceil(t_end * rps)，到达时间 t = 0, 1/rps, 2/rps, ..., (N-1)/rps
    - random 数据集：一次性 rng.choices(template, weights, k=N) 预采样，与 benchmark 一致

    数据集来源由 simulation.dataset 决定：
      - trace  : 使用 trace_path 提供的 size/steps 序列；arrival_time 仍按 rps 均匀注入
      - default: 使用 default_request 中配置的单一请求类型
      - random : 使用 random_request_config，按 weight 预采样 N 个（与 benchmark RandomDataset 一致）
    """
    sim = cfg["simulation"]
    default = cfg.get("default_request", {})
    size_default = default.get("size", "128x128")
    if "height" in default and "width" in default:
        height_default = int(default.get("height"))
        width_default = int(default.get("width"))
    else:
        height_default, width_default = _parse_size(size_default)
    num_frames_default = int(default.get("num_frames", 1))
    task_type_default = str(default.get("task_type") or ("video" if num_frames_default > 1 else "image")).lower().strip()
    steps_default = int(default.get("steps", 5))
    dataset = str(sim.get("dataset", "") or "").strip().lower()
    trace_path = sim.get("trace_path")

    template: list[dict]
    weights: list[float] | None = None
    rng: random.Random | None = None

    if dataset == "random":
        random_cfg = sim.get("random_request_config") or []
        if not isinstance(random_cfg, list) or not random_cfg:
            raise ValueError("dataset=random 但 random_request_config 为空或不是列表")
        template = []
        weights = []
        for item in random_cfg:
            if not isinstance(item, dict):
                continue
            w = int(item.get("width", width_default))
            h = int(item.get("height", height_default))
            nf = int(item.get("num_frames", num_frames_default))
            tt = str(item.get("task_type") or ("video" if nf > 1 else "image")).lower().strip()
            st = int(item.get("steps", item.get("num_inference_steps", steps_default)))
            wt = float(item.get("weight", 1.0))
            if wt <= 0:
                continue
            template.append({
                "size": f"{h}x{w}",
                "height": h,
                "width": w,
                "num_frames": nf,
                "task_type": tt,
                "steps": st,
            })
            weights.append(wt)
        if not template:
            raise ValueError("dataset=random 但 random_request_config 解析后为空")
        seed = int(sim.get("random_request_seed", 42))
        rng = random.Random(seed)
    else:
        # 兼容旧行为：未显式配置 dataset 时，若 trace_path 存在则使用 trace，否则使用 default_request。
        if (not dataset or dataset == "trace") and trace_path and Path(trace_path).exists():
            template = load_trace_template(Path(trace_path))
            if not template:
                raise ValueError(f"trace 解析后无有效请求: {trace_path}")
        elif dataset in ("trace", "", None):
            template = [{
                "size": f"{height_default}x{width_default}",
                "height": height_default,
                "width": width_default,
                "num_frames": num_frames_default,
                "task_type": task_type_default,
                "steps": steps_default,
            }]
        elif dataset == "default":
            template = [{
                "size": f"{height_default}x{width_default}",
                "height": height_default,
                "width": width_default,
                "num_frames": num_frames_default,
                "task_type": task_type_default,
                "steps": steps_default,
            }]
        else:
            raise ValueError(f"不支持的 dataset 类型: {dataset!r}，可选: trace, default, random")

    # 与 benchmark 一致：N = ceil(t_end * rps)，到达时间 t = 0, 1/rps, ..., (N-1)/rps
    n_requests = max(1, math.ceil(t_end * rps)) if rps > 0 else 0
    if dataset == "random" and rng is not None and weights is not None:
        sampled = rng.choices(template, weights=weights, k=n_requests)
    else:
        sampled = None

    requests = []
    for i in range(n_requests):
        t = i / rps if rps > 0 else 0.0
        base_req = sampled[i] if sampled is not None else template[i % len(template)]
        req = base_req.copy()
        req["request_id"] = f"req_{i}"
        req["arrival_time"] = t
        requests.append(req)
    return requests


def assign_slo(requests: list[dict], profile_data: dict, workers: list[dict], slo_scale: float) -> None:
    """为每个请求赋 slo_ms：用第一个 worker 的 profile 基准 * 1000 * slo_scale。"""
    if not workers or slo_scale <= 0:
        return
    first_worker = workers[0]
    worker_ref = {
        "config_id": worker_config_id(first_worker),
        "instance_type": first_worker.get("instance_type"),
    }
    for r in requests:
        st = lookup(profile_data, r, worker_ref)
        r["slo_ms"] = st * 1000.0 * slo_scale


# ---------- 调度模块（可插拔） ----------

def dispatch_round_robin(request: dict, workers: list, state: dict) -> int:
    """轮询：与 vllm_omni.global_scheduler.policies.round_robin.RoundRobinPolicy 对齐。

    维护实例列表上的游标，每次从游标位置起按索引顺序选取首个有效实例（若有空闲则仅在空闲中选，
    若全忙则在全体中选）。游标按实例顺序推进，保证与 global_scheduler 一致。
    """
    del request
    n = len(workers)
    if n == 0:
        raise ValueError("No workers configured")
    available = [i for i in range(n) if workers[i]["next_time"] == INF]
    candidates = available if available else list(range(n))
    cursor = state.setdefault("next_index", 0)
    start = cursor % n
    selected = start
    if start not in candidates:
        for offset in range(1, n):
            probe = (start + offset) % n
            if probe in candidates:
                selected = probe
                break
    state["next_index"] = (selected + 1) % n
    return selected


def dispatch_min_queue_length(request: dict, workers: list, state: dict) -> int:
    """最小队列长度：优先在空闲实例中选择队列长度（不含正在运行请求）最小者；若全部忙碌，则在全体实例中选队列长度最小者。"""
    del request, state
    n = len(workers)
    if n == 0:
        raise ValueError("No workers configured")
    available = [i for i, w in enumerate(workers) if w["next_time"] == INF]
    indices = available if available else list(range(n))
    best = indices[0]
    best_len = len(workers[best]["queue"])
    for i in indices[1:]:
        qlen = len(workers[i]["queue"])
        if qlen < best_len:
            best_len = qlen
            best = i
    return best


def _queued_work_s(worker: dict, profile_data: dict, current_time: float) -> float:
    """该 worker 队列中的剩余总工作量（秒），不包含当前正在运行请求。查不到配置即报错（fail-fast）。"""
    del current_time
    queued = 0.0
    for r in worker["queue"]:
        queued += lookup(profile_data, r, worker)
    return queued


def dispatch_short_queue_runtime(request: dict, workers: list, state: dict) -> int:
    """最短队列预估时间：选当前队列中预估总工作量（秒）最小的实例（不计正在运行请求）。"""
    best = 0
    best_work = INF
    for i, w in enumerate(workers):
        work = w.get("_queued_work")
        if work is None:
            work = _queued_work_s(w, state["profile_data"], state.get("current_time", 0.0))
        if work < best_work:
            best_work = work
            best = i
    return best


DISPATCH = {
    "round_robin": dispatch_round_robin,
    "min_queue_length": dispatch_min_queue_length,
    "short_queue_runtime": dispatch_short_queue_runtime,
}

def _pop_next_from_queue(w: dict, profile_data: dict, instance_policy: str) -> dict | None:
    """从 worker 队列中取下一个待执行请求。fcfs=队首，sjf=服务时间最短。"""
    if not w["queue"]:
        return None
    if instance_policy == "sjf":
        best_i = 0
        best_st = lookup(profile_data, w["queue"][0], w)
        for i in range(1, len(w["queue"])):
            st = lookup(profile_data, w["queue"][i], w)
            if st < best_st:
                best_st = st
                best_i = i
        return w["queue"].pop(best_i)
    return w["queue"].pop(0)


def run_simulation(
    requests: list[dict],
    workers: list[dict],
    profile_data: dict,
    rps: float,
    t_end: float,
    algorithm: str,
    instance_scheduler_policy: str = "fcfs",
) -> dict:
    """单次模拟，返回与 plot_results 对齐的指标 + algorithm。

    instance_scheduler_policy: fcfs（队首）或 sjf（最短作业优先），与 benchmark --instance-scheduler-policy 对齐。
    """
    if algorithm not in DISPATCH:
        raise ValueError(f"不支持的算法: {algorithm}，可选: {list(DISPATCH.keys())}")
    dispatch_fn = DISPATCH[algorithm]
    state = {"profile_data": profile_data}

    # 复制 worker 状态：config_id, queue, next_time
    ws = []
    for w in workers:
        cid = worker_config_id(w)
        wr = {
            "config_id": cid,
            "instance_type": w.get("instance_type", cid),
            "queue": [],
            "next_time": INF,
        }
        if algorithm == "short_queue_runtime":
            wr["_queued_work"] = 0.0  # 增量维护队列总工作量，避免 O(n^2)
        ws.append(wr)
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
                if algorithm == "short_queue_runtime":
                    st = lookup(profile_data, req, ws[wi])
                    req["_svc"] = st
                    ws[wi]["_queued_work"] += st
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

        # 立即派工：空闲 worker 从队列取请求，fcfs=队首/sjf=最短服务时间
        for w in ws:
            if w["next_time"] == INF and w["queue"]:
                req = _pop_next_from_queue(w, profile_data, instance_scheduler_policy)
                if algorithm == "short_queue_runtime":
                    w["_queued_work"] -= req.get("_svc", 0.0)
                req["start_time"] = t  # 从本时刻开始执行
                st = lookup(profile_data, req, w)
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
        n_in_window = 0
        latency_mean = latency_median = latency_p50 = latency_p95 = latency_p99 = 0.0
        waiting_time_mean = waiting_time_p95 = waiting_time_p99 = 0.0
        service_time_mean = service_time_p95 = service_time_p99 = 0.0
        slo_attainment_rate = 0.0
    else:
        t_first = min(r["arrival_time"] for r in completed)
        t_last = max(r["finish_time"] for r in completed)
        duration = t_last - t_first
        completed_in_window = [r for r in completed if r["finish_time"] <= t_end]
        n_in_window = len(completed_in_window)
        # 吞吐量 = 总完成请求数 / 总持续时间（首请求到达至末请求完成的时长），与 benchmark 一致
        throughput_qps = n_ok / duration if duration > 0 else 0.0
        # 口径对齐：latency/waiting/service/slo 基于全部已完成请求，与 diffusion_benchmark_serving 一致
        stats_base = completed
        latencies = [r["latency"] for r in stats_base if "latency" in r]
        waiting_times = [r["waiting_time"] for r in stats_base if "waiting_time" in r]
        service_times = [r["service_time"] for r in stats_base if "service_time" in r]
        latencies.sort()
        n_stats = len(latencies)
        latency_mean = sum(latencies) / n_stats if n_stats else 0.0
        latency_median = float(np.median(latencies)) if latencies else 0.0
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
        slo_met = sum(1 for r in stats_base if r.get("slo_ms") is not None and r["latency"] * 1000 <= r["slo_ms"])
        slo_total = sum(1 for r in stats_base if r.get("slo_ms") is not None)
        slo_attainment_rate = (slo_met / slo_total) if slo_total else 0.0

    return {
        "algorithm": algorithm,
        "rps": rps,
        "duration": round(duration, 4),
        "completed_requests": n_ok,
        "completed_in_window": n_in_window,
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
    parser.add_argument("config", nargs="?", default="config/simulation_config.yaml", help="YAML 配置文件路径")
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

    try:
        profile_data = load_profile(profile_path)
    except (ValueError, KeyError) as e:
        print(f"profile 加载失败: {e}", file=sys.stderr)
        sys.exit(1)
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
            assign_slo(requests, profile_data, workers, slo_scale)
            try:
                instance_policy = str(cfg.get("scheduler", {}).get("instance_scheduler_policy", "fcfs")).strip().lower()
                out = run_simulation(requests, workers, profile_data, rps, t_end, algo, instance_scheduler_policy=instance_policy)
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
        plot_output_dir = plot_cfg.get("output_dir")
        if plot_output_dir in (None, "output", "../output"):
            plot_output_dir = output_dir  # 与仿真结果同目录：output/{profile名}/
        else:
            base = config_path.parent
            if not Path(plot_output_dir).is_absolute():
                plot_output_dir = base / plot_output_dir
            else:
                plot_output_dir = Path(plot_output_dir)
        plot_output_dir = Path(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        plot_output_prefix = plot_cfg.get("output_prefix", "compare")
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
