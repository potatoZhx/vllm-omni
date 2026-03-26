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

# ---------- 实例调度器常量（与 stage1_scheduler.py 对齐） ----------
_SJF_AGING_DEFAULT_FACTOR = 1.0
_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS = (512, 768, 1024)
_SIZE_BUCKET_PROMOTION_WINDOW_S = 10.0
_P95_FIRST_HISTORY_MAXLEN = 128
_P95_FIRST_MIN_HISTORY_FOR_QUANTILE = 20
_P95_FIRST_SERVICE_RATE_EMA_ALPHA = 0.1
_SJF_AGING_COST_REF_S = 12.0
_SJF_AGING_COST_WEIGHT_MAX = 4.0
_SJF_AGING_GUARDED_MIN_WAIT_S = 45.0
_SJF_AGING_GUARDED_WAIT_COST_RATIO = 2.0
_P95_BUCKET_DEFAULT_COUNT = 4
_P95_BUCKET_DEFAULT_MIN_WINDOW_MS = 200.0


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
        # output_dir 为 output 根目录时，自动追加 profile 名 + 算法 + 实例策略：output/{profile名}_{算法}_{策略}/
        # 笛卡尔乘积时：output/{profile名}_{算法1_算法2}_{fcfs_sjf}/
        if out.name == "output":
            sched = cfg.get("scheduler", {})
            workers = cfg.get("workers", [])
            n_workers = len(workers) if workers else 0
            algorithms_to_run = sched.get("algorithms_to_run", None)
            if isinstance(algorithms_to_run, str):
                algorithms_to_run = [algorithms_to_run]
            policies_raw = sched.get("instance_scheduler_policies") or sched.get("instance_scheduler_policy", "fcfs")
            instance_policies = (
                [str(p).strip().lower() for p in policies_raw]
                if isinstance(policies_raw, (list, tuple))
                else [str(policies_raw).strip().lower()]
            )
            parts = [profile_stem]
            if n_workers > 0:
                parts.append(f"n{n_workers}")
            if algorithms_to_run:
                parts.append("_".join(sorted(algorithms_to_run)))
            if instance_policies:
                parts.append("_".join(sorted(instance_policies)))
            out = out / "_".join(parts)
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
    """最小队列长度：选 inflight+queue_len 最小的实例，与 instance_local_schd MinQueueLengthPolicy 对齐。"""
    del request, state
    n = len(workers)
    if n == 0:
        raise ValueError("No workers configured")
    available = [i for i, w in enumerate(workers) if w["next_time"] == INF]
    indices = available if available else list(range(n))
    best = indices[0]
    inflight = 1 if workers[best].get("current_request") else 0
    best_score = len(workers[best]["queue"]) + inflight
    for i in indices[1:]:
        inflight_i = 1 if workers[i].get("current_request") else 0
        score = len(workers[i]["queue"]) + inflight_i
        if score < best_score:
            best_score = score
            best = i
    return best


def _queued_work_s(worker: dict, profile_data: dict, current_time: float) -> float:
    """该 worker 队列中的剩余总工作量（秒），不包含当前正在运行请求。对已抢占请求用 _remaining_latency_s。"""
    del current_time
    queued = 0.0
    for r in worker["queue"]:
        queued += _remaining_latency_s(r, profile_data, worker)
    return queued


def _total_outstanding_work_s(
    w: dict, profile_data: dict, state: dict
) -> float:
    """worker 总未完成工作量（秒）= 队列剩余 + 正在运行请求的剩余。与 instance_local_schd ShortQueueRuntimePolicy 对齐。"""
    base = w.get("_queued_work")
    if base is None:
        base = _queued_work_s(w, profile_data, state.get("current_time", 0.0))
    cur = w.get("current_request")
    if cur is not None:
        base += _remaining_latency_s(cur, profile_data, w)
    return base


def dispatch_short_queue_runtime(request: dict, workers: list, state: dict) -> int:
    """最短队列预估时间：选总未完成工作量（队列+inflight 剩余）最小的实例，与 instance_local_schd 对齐。"""
    best = 0
    best_work = INF
    for i, w in enumerate(workers):
        work = _total_outstanding_work_s(w, state["profile_data"], state)
        if work < best_work:
            best_work = work
            best = i
    return best


def _max_service_time(profile_data: dict) -> float:
    """从 profile 中取最大服务时间（用于识别最大类作业）。"""
    return max(profile_data["json_table"].values())


def dispatch_short_queue_runtime_max_class_balanced(request: dict, workers: list, state: dict) -> int:
    """short_queue_runtime + 最大类作业均衡：最大类作业强制轮转分布到各 worker，其余同 short_queue_runtime。

    最大类 = profile 中 service_time 最大的请求类型（如 1536×1536）。均衡策略：选当前最大类作业数最少的 worker；
    同数时按 short_queue_runtime 选队列工作量最小的。
    """
    profile_data = state["profile_data"]
    max_st = state.setdefault("_max_st", _max_service_time(profile_data))
    ref = workers[0] if workers else {}
    is_max = lookup(profile_data, request, ref) >= max_st * 0.99

    if not is_max:
        return dispatch_short_queue_runtime(request, workers, state)

    # 最大类：选 _max_class_count 最小的；同数时按总未完成工作量（队列+inflight）最小
    best = 0
    best_count = workers[0].get("_max_class_count", 0)
    best_work = _total_outstanding_work_s(workers[0], profile_data, state)
    for i, w in enumerate(workers):
        cnt = w.get("_max_class_count", 0)
        work = _total_outstanding_work_s(w, profile_data, state)
        if cnt < best_count or (cnt == best_count and work < best_work):
            best_count = cnt
            best_work = work
            best = i
    return best


DISPATCH = {
    "round_robin": dispatch_round_robin,
    "min_queue_length": dispatch_min_queue_length,
    "short_queue_runtime": dispatch_short_queue_runtime,
    "short_queue_runtime_max_class_balanced": dispatch_short_queue_runtime_max_class_balanced,
}


def _remaining_latency_s(request: dict, profile_data: dict, worker: dict) -> float:
    """请求剩余预估服务时间（秒）。未开始则返回全量；已执行部分则按 steps 比例估算。"""
    total_s = lookup(profile_data, request, worker)
    total_steps = max(int(request.get("steps", 1) or 1), 1)
    executed = int(request.get("_executed_steps", 0) or 0)
    remaining_steps = max(total_steps - executed, 0)
    if remaining_steps <= 0:
        return 0.0
    return total_s * remaining_steps / total_steps


def _heuristic_cost_s(request: dict) -> float:
    """Heuristic cost = steps * num_frames * area_scale.

    Matches stage1_scheduler._estimate_cost_seconds when no runtime profile
    is configured (the profiled_estimate falls back to the heuristic).
    Numerically identical to _work_units.
    """
    return max(_work_units(request), 1e-9)


def _scheduling_cost_s(request: dict, profile_data: dict, worker: dict) -> float:
    """Cost estimate for scheduling sort keys.

    When worker["_use_heuristic_cost"] is True, returns the area-based
    heuristic (matching real deployments without --diffusion-runtime-profile).
    Otherwise returns the profile-based remaining latency.
    """
    if worker.get("_use_heuristic_cost"):
        return _heuristic_cost_s(request)
    return max(_remaining_latency_s(request, profile_data, worker), 1e-9)


def _chunk_duration_s(
    request: dict,
    profile_data: dict,
    worker: dict,
    chunk_budget_steps: int,
    small_request_threshold_ms: float | None = None,
) -> tuple[float, int]:
    """本次 chunk 预估耗时（秒）与本次执行的 steps 数。返回 (duration_s, chunk_steps)。

    若 small_request_threshold_ms 已设置且剩余预估时间 <= 阈值，则直接跑完剩余 steps，不再切 chunk。
    与 instance_local_schd 的 diffusion_small_request_latency_threshold_ms 对应。
    """
    remaining_s = _remaining_latency_s(request, profile_data, worker)
    total_steps = max(int(request.get("steps", 1) or 1), 1)
    executed = int(request.get("_executed_steps", 0) or 0)
    remaining_steps = max(total_steps - executed, 0)
    if remaining_steps <= 0:
        return 0.0, 0
    # 小请求优化：剩余时间过短则直接跑完，不再抢占
    if small_request_threshold_ms is not None and small_request_threshold_ms > 0:
        if remaining_s * 1000.0 <= float(small_request_threshold_ms):
            return remaining_s, remaining_steps
    chunk_steps = min(chunk_budget_steps, remaining_steps)
    duration = remaining_s * chunk_steps / remaining_steps
    return duration, chunk_steps


def _sjf_preempt_budget_steps_for_request(
    request: dict,
    default_budget: int,
    image_budget: int | None,
    video_budget: int | None,
) -> int:
    """与 diffusion_engine._chunk_budget_steps_for_request 一致：仅 num_frames>1 用 video_budget，否则 image_budget；缺省回退 default。"""
    num_frames = max(int(request.get("num_frames", 1) or 1), 1)
    if num_frames > 1:
        b = video_budget if video_budget is not None else default_budget
    else:
        b = image_budget if image_budget is not None else default_budget
    return max(int(b), 1)


def _policy_uses_chunk_preempt(policy: str) -> bool:
    """判断实例调度策略是否使用 chunk 抢占。

    - *_no_preempt 后缀: 强制不抢占（用于 sjf_aging_no_preempt 等）
    - sjf_chunk_preempt / sjf_preempt: 显式抢占
    - sjf_aging / sjf_aging_guarded / size_bucket_sjf_aging: 默认启用
    - p95-first / p95-bucket-sjf / p95-bucket-sjf-normalized: 默认启用
    - fcfs / sjf: 不抢占
    """
    if policy.endswith("_no_preempt"):
        return False
    if policy.startswith("sjf_chunk_preempt") or policy.startswith("sjf_preempt"):
        return True
    if policy.startswith("sjf_aging") or policy.startswith("size_bucket_sjf_aging"):
        return True
    if policy.startswith("p95"):
        return True
    return False


def _parse_aging_factor(policy: str) -> float:
    """从策略名解析 aging factor。与 stage1_scheduler._effective_sjf_aging_factor 对齐。

    sjf_aging -> 1.0, sjf_aging_0.15 -> 0.15,
    size_bucket_sjf_aging -> 1.0, size_bucket_sjf_aging_0.5 -> 0.5
    """
    for prefix in ("size_bucket_sjf_aging", "sjf_aging"):
        if policy.startswith(prefix):
            rest = policy[len(prefix):]
            if rest.startswith("_"):
                try:
                    return float(rest[1:])
                except ValueError:
                    pass
            return _SJF_AGING_DEFAULT_FACTOR
    return _SJF_AGING_DEFAULT_FACTOR


def _sjf_aging_cost_weight(estimated_cost_s: float) -> float:
    """与 stage1_scheduler._sjf_aging_cost_weight 对齐。
    大请求 aging 更快: weight = clamp(sqrt(cost / 12.0), 1.0, 4.0)
    """
    normalized_cost = max(float(estimated_cost_s), 1e-9) / _SJF_AGING_COST_REF_S
    return min(max(math.sqrt(normalized_cost), 1.0), _SJF_AGING_COST_WEIGHT_MAX)


def _sjf_aging_sort_key(
    request: dict, profile_data: dict, worker: dict,
    aging_factor: float, current_time: float,
) -> tuple:
    """sjf_aging 排序键。与 stage1_scheduler._build_sjf_aging_queue 对齐。

    aged_cost = estimated_cost / (1 + aging_factor * cost_weight * age)
    sort by (aged_cost, estimated_cost, enqueue_time)
    """
    estimated_cost = _scheduling_cost_s(request, profile_data, worker)
    age_s = max(current_time - request.get("arrival_time", 0.0), 0.0)
    cost_weight = _sjf_aging_cost_weight(estimated_cost)
    aged_cost = estimated_cost / (1.0 + aging_factor * cost_weight * age_s)
    return (aged_cost, estimated_cost, request.get("arrival_time", 0.0))


def _sjf_aging_guarded_sort_key(
    request: dict, profile_data: dict, worker: dict,
    aging_factor: float, current_time: float,
) -> tuple:
    """sjf_aging_guarded 排序键。与 stage1_scheduler._build_sjf_aging_guarded_queue 对齐。

    同 sjf_aging（含 cost_weight）+ 尾部保护：
    等待过久的请求 (age >= max(45s, 2*cost)) 升级为 protected 组，组内 FIFO。
    """
    estimated_cost = _scheduling_cost_s(request, profile_data, worker)
    age_s = max(current_time - request.get("arrival_time", 0.0), 0.0)
    cost_weight = _sjf_aging_cost_weight(estimated_cost)
    aged_cost = estimated_cost / (1.0 + aging_factor * cost_weight * age_s)
    protection_threshold_s = max(
        _SJF_AGING_GUARDED_MIN_WAIT_S,
        _SJF_AGING_GUARDED_WAIT_COST_RATIO * estimated_cost,
    )
    tail_protected = age_s >= protection_threshold_s
    if tail_protected:
        return (0, request.get("arrival_time", 0.0), request.get("arrival_time", 0.0))
    return (1, aged_cost, estimated_cost, request.get("arrival_time", 0.0))


def _fixed_size_bucket_id(request: dict) -> int:
    """按 max(width, height) 分桶。与 stage1_scheduler._fixed_size_bucket_id 对齐。"""
    width = int(request.get("width", 1024))
    height = int(request.get("height", 1024))
    max_dim = max(width, height)
    for bucket_id, threshold in enumerate(_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS):
        if max_dim <= threshold:
            return bucket_id
    return len(_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS)


def _size_bucket_sjf_aging_sort_key(
    request: dict, profile_data: dict, worker: dict,
    aging_factor: float, current_time: float,
) -> tuple:
    """size_bucket_sjf_aging 排序键。与 stage1_scheduler._build_size_bucket_sjf_aging_queue 对齐。

    1. 按 max(w,h) 分桶 (512, 768, 1024 阈值)
    2. bucket 内按 estimated_cost / (1 + aging_factor * age) 排序
    3. 等待过久时 bucket 晋升：promotion_levels = int(aging_factor * age / 10.0)
    """
    estimated_cost = _scheduling_cost_s(request, profile_data, worker)
    age_s = max(current_time - request.get("arrival_time", 0.0), 0.0)
    aged_cost = estimated_cost / (1.0 + aging_factor * age_s)
    raw_bucket = _fixed_size_bucket_id(request)
    promotion_levels = int((aging_factor * age_s) / _SIZE_BUCKET_PROMOTION_WINDOW_S)
    effective_bucket = max(raw_bucket - promotion_levels, 0)
    return (effective_bucket, aged_cost, estimated_cost, request.get("arrival_time", 0.0))


def _work_units(request: dict) -> float:
    """计算 work_units。与 stage1_scheduler._request_work_units 对齐。

    work_units = remaining_steps * num_frames * max(area / 1024^2, 0.0625)
    """
    total_steps = max(int(request.get("steps", 1) or 1), 1)
    executed = int(request.get("_executed_steps", 0) or 0)
    remaining_steps = max(total_steps - executed, 0)
    if remaining_steps <= 0:
        return 1e-9
    width = int(request.get("width", 1024))
    height = int(request.get("height", 1024))
    num_frames = max(int(request.get("num_frames", 1) or 1), 1)
    area_scale = max((width * height) / (1024 * 1024), 0.0625)
    return max(float(remaining_steps * num_frames) * area_scale, 1e-9)


def _p95_learned_slowdown(worker: dict) -> float:
    """计算 learned slowdown p95。与 stage1_scheduler._learned_p95_first_slowdown 对齐。"""
    history = worker.get("_p95_slowdown_history")
    if not history:
        return 1.0
    samples = sorted(history)
    if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
        return max(float(samples[-1]), 1.0)
    index = max(math.ceil(len(samples) * 0.95) - 1, 0)
    return max(float(samples[index]), 1.0)


def _p95_learned_ms(worker: dict) -> float:
    """从 worker 的绝对延迟历史计算 P95。与 stage1_scheduler._learned_p95_ms 对齐。"""
    history = worker.get("_p95_latency_history_ms")
    if not history:
        return 0.0
    samples = sorted(history)
    if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
        return float(samples[-1])
    index = max(math.ceil(len(samples) * 0.95) - 1, 0)
    return float(samples[index])


def _p95_record_latency_ms(worker: dict, latency_ms: float) -> None:
    """记录完成请求的绝对延迟 ms。与 stage1_scheduler._record_p95_first_latency_ms 对齐。"""
    history = worker.setdefault("_p95_latency_history_ms", [])
    history.append(max(float(latency_ms), 0.0))
    if len(history) > _P95_FIRST_HISTORY_MAXLEN:
        worker["_p95_latency_history_ms"] = history[-_P95_FIRST_HISTORY_MAXLEN:]


def _p95_estimated_service_ms(request: dict, profile_data: dict, worker: dict) -> float:
    """估计服务时间（ms）。与 stage1_scheduler._p95_first_estimated_service_ms 对齐。

    若有在线学习的 service_rate，使用 rate * work_units；否则回退到 scheduling cost。
    """
    rate = worker.get("_p95_service_rate_ms_per_wu")
    if rate is not None:
        wu = _work_units(request)
        return max(rate * wu, 1e-9)
    return max(_scheduling_cost_s(request, profile_data, worker) * 1000.0, 1e-9)


def _p95_record_chunk_sample(worker: dict, chunk_duration_s: float, request: dict) -> None:
    """记录 chunk 执行样本，更新 service rate EMA。与 stage1_scheduler._record_p95_first_execute_sample 对齐。"""
    chunk_wu = _work_units_for_chunk(request)
    if chunk_duration_s <= 0 or chunk_wu <= 0:
        return
    observed_rate = (chunk_duration_s * 1000.0) / max(chunk_wu, 1e-9)
    if observed_rate <= 0:
        return
    current = worker.get("_p95_service_rate_ms_per_wu")
    if current is None:
        worker["_p95_service_rate_ms_per_wu"] = observed_rate
    else:
        alpha = _P95_FIRST_SERVICE_RATE_EMA_ALPHA
        worker["_p95_service_rate_ms_per_wu"] = (1.0 - alpha) * current + alpha * observed_rate


def _work_units_for_chunk(request: dict) -> float:
    """计算本次 chunk 的 work_units（基于 _chunk_steps）。"""
    chunk_steps = int(request.get("_chunk_steps", 0) or 0)
    if chunk_steps <= 0:
        return 0.0
    width = int(request.get("width", 1024))
    height = int(request.get("height", 1024))
    num_frames = max(int(request.get("num_frames", 1) or 1), 1)
    area_scale = max((width * height) / (1024 * 1024), 0.0625)
    return max(float(chunk_steps * num_frames) * area_scale, 1e-9)


def _p95_record_completion(worker: dict, latency_s: float, total_execute_s: float) -> None:
    """记录完成请求的 slowdown，用于 p95 学习。与 stage1_scheduler._record_p95_first_latency_ms 对齐。"""
    if total_execute_s <= 0:
        return
    slowdown = max(latency_s / max(total_execute_s, 1e-9), 1.0)
    history = worker.setdefault("_p95_slowdown_history", [])
    history.append(slowdown)
    if len(history) > _P95_FIRST_HISTORY_MAXLEN:
        worker["_p95_slowdown_history"] = history[-_P95_FIRST_HISTORY_MAXLEN:]


def _pop_p95_first(w: dict, profile_data: dict, current_time: float) -> dict | None:
    """p95-first greedy 选择下一个请求。与 stage1_scheduler._build_p95_first_queue 的首轮选择对齐。

    worker 空闲时 cursor_ms = 0, active_chunk_blocking_ms = 0。
    对每个候选计算 pressure_ratio = predicted_finish_ms / target_latency_ms，
    选 final_priority_score (= -pressure_ratio) 最小（即 pressure 最大）的。
    """
    queue = w["queue"]
    if not queue:
        return None
    learned_slowdown = _p95_learned_slowdown(w)
    cursor_ms = 0.0
    best_i = 0
    best_key = None
    for i, r in enumerate(queue):
        est_service_ms = _p95_estimated_service_ms(r, profile_data, w)
        age_s = max(current_time - r.get("arrival_time", 0.0), 0.0)
        predicted_finish_ms = max(age_s * 1000.0 + cursor_ms + est_service_ms, 0.0)
        target_latency_ms = max(learned_slowdown * est_service_ms, 1e-9)
        pressure_ratio = predicted_finish_ms / target_latency_ms
        final_score = -pressure_ratio
        key = (final_score, -pressure_ratio, est_service_ms, r.get("arrival_time", 0.0), r.get("request_id", ""))
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return queue.pop(best_i)


def _pop_p95_bucket_sjf(w: dict, profile_data: dict, current_time: float) -> dict | None:
    """p95-bucket-sjf: 按历史 P95 绝对延迟分桶 + 桶内 SJF。
    与 stage1_scheduler._build_p95_bucket_sjf_queue 对齐。
    """
    queue = w["queue"]
    if not queue:
        return None
    bucket_count = _P95_BUCKET_DEFAULT_COUNT
    min_window_ms = _P95_BUCKET_DEFAULT_MIN_WINDOW_MS
    availability_ts = current_time

    history_p95_ms = max(_p95_learned_ms(w), min_window_ms)
    max_estimated_cost_ms = max(
        _scheduling_cost_s(r, profile_data, w) * 1000.0 for r in queue
    )
    anchor_window_ms = max(history_p95_ms, max_estimated_cost_ms, min_window_ms)
    bucket_width_ms = max(anchor_window_ms / float(bucket_count), 1e-9)

    best_i = 0
    best_key = None
    for i, r in enumerate(queue):
        cost_s = _scheduling_cost_s(r, profile_data, w)
        cost_ms = cost_s * 1000.0
        target_p95_ms = max(history_p95_ms, cost_ms)
        arrival = r.get("arrival_time", 0.0)
        deadline_ts = arrival + (target_p95_ms / 1000.0)
        urgency_ms = (deadline_ts - availability_ts) * 1000.0
        bucket_id = 0 if urgency_ms <= 0.0 else min(
            int(urgency_ms / bucket_width_ms), bucket_count - 1,
        )
        key = (bucket_id, cost_s, arrival, r.get("request_id", ""))
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return queue.pop(best_i)


def _pop_p95_bucket_sjf_normalized(
    w: dict, profile_data: dict, current_time: float,
) -> dict | None:
    """p95-bucket-sjf-normalized: 按归一化 target_latency 分桶 + 桶内按 service_ms 排。
    与 stage1_scheduler._build_p95_bucket_sjf_normalized_queue 对齐。
    """
    queue = w["queue"]
    if not queue:
        return None
    bucket_count = _P95_BUCKET_DEFAULT_COUNT
    min_window_ms = _P95_BUCKET_DEFAULT_MIN_WINDOW_MS
    availability_ts = current_time

    learned_slowdown = _p95_learned_slowdown(w)

    max_target_latency_ms = min_window_ms
    estimated_services: list[float] = []
    for r in queue:
        est_svc = _p95_estimated_service_ms(r, profile_data, w)
        target_latency_ms = max(learned_slowdown * est_svc, 1e-9)
        max_target_latency_ms = max(max_target_latency_ms, target_latency_ms)
        estimated_services.append(est_svc)

    bucket_width_ms = max(max_target_latency_ms / float(bucket_count), 1e-9)

    best_i = 0
    best_key = None
    for i, r in enumerate(queue):
        est_svc = estimated_services[i]
        target_latency_ms = max(learned_slowdown * est_svc, 1e-9)
        arrival = r.get("arrival_time", 0.0)
        synthetic_deadline_ts = arrival + (target_latency_ms / 1000.0)
        urgency_ms = (synthetic_deadline_ts - availability_ts) * 1000.0
        bucket_id = 0 if urgency_ms <= 0.0 else min(
            int(urgency_ms / bucket_width_ms), bucket_count - 1,
        )
        key = (bucket_id, est_svc, arrival, r.get("request_id", ""))
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return queue.pop(best_i)


def _pop_next_from_queue(
    w: dict, profile_data: dict, instance_policy: str, current_time: float = 0.0,
) -> dict | None:
    """从 worker 队列中取下一个待执行请求。与 stage1_scheduler.py 各策略对齐。

    _no_preempt 后缀仅影响抢占开关，不影响队列排序（在此剥离）。

    fcfs: 队首 (FIFO)
    sjf: 按总服务时间最短优先（无 chunk 抢占）
    sjf_chunk_preempt / sjf_preempt: 按剩余服务时间最短优先 (SRPT)，chunk 边界抢占
    sjf_aging: remaining_cost / (1 + aging_factor * cost_weight * age) 排序
    sjf_aging_guarded: sjf_aging + 尾部保护（等待过久 FIFO 优先）
    size_bucket_sjf_aging: 先按分辨率分桶，桶内 sjf_aging 排序
    p95-first: tail pressure greedy 排序
    p95-bucket-sjf: 按历史 P95 分桶 + 桶内 SJF
    p95-bucket-sjf-normalized: 按归一化 target_latency 分桶 + 桶内按 service_ms
    """
    if not w["queue"]:
        return None

    base = instance_policy.removesuffix("_no_preempt")

    if base == "fcfs":
        return w["queue"].pop(0)

    if base == "sjf":
        best_i = 0
        best_st = lookup(profile_data, w["queue"][0], w)
        for i in range(1, len(w["queue"])):
            st = lookup(profile_data, w["queue"][i], w)
            if st < best_st or (st == best_st and w["queue"][i].get("arrival_time", 0) < w["queue"][best_i].get("arrival_time", 0)):
                best_st = st
                best_i = i
        return w["queue"].pop(best_i)

    if base.startswith("sjf_chunk_preempt") or base.startswith("sjf_preempt"):
        best_i = 0
        best_rem = _remaining_latency_s(w["queue"][0], profile_data, w)
        for i in range(1, len(w["queue"])):
            rem = _remaining_latency_s(w["queue"][i], profile_data, w)
            if rem < best_rem or (rem == best_rem and w["queue"][i].get("arrival_time", 0) < w["queue"][best_i].get("arrival_time", 0)):
                best_rem = rem
                best_i = i
        return w["queue"].pop(best_i)

    if base.startswith("sjf_aging_guarded"):
        aging_factor = _parse_aging_factor(base)
        best_i = 0
        best_key = _sjf_aging_guarded_sort_key(w["queue"][0], profile_data, w, aging_factor, current_time)
        for i in range(1, len(w["queue"])):
            key = _sjf_aging_guarded_sort_key(w["queue"][i], profile_data, w, aging_factor, current_time)
            if key < best_key:
                best_key = key
                best_i = i
        return w["queue"].pop(best_i)

    if base.startswith("sjf_aging"):
        aging_factor = _parse_aging_factor(base)
        best_i = 0
        best_key = _sjf_aging_sort_key(w["queue"][0], profile_data, w, aging_factor, current_time)
        for i in range(1, len(w["queue"])):
            key = _sjf_aging_sort_key(w["queue"][i], profile_data, w, aging_factor, current_time)
            if key < best_key:
                best_key = key
                best_i = i
        return w["queue"].pop(best_i)

    if base.startswith("size_bucket_sjf_aging"):
        aging_factor = _parse_aging_factor(base)
        best_i = 0
        best_key = _size_bucket_sjf_aging_sort_key(w["queue"][0], profile_data, w, aging_factor, current_time)
        for i in range(1, len(w["queue"])):
            key = _size_bucket_sjf_aging_sort_key(w["queue"][i], profile_data, w, aging_factor, current_time)
            if key < best_key:
                best_key = key
                best_i = i
        return w["queue"].pop(best_i)

    if base.startswith("p95-bucket-sjf-normalized"):
        return _pop_p95_bucket_sjf_normalized(w, profile_data, current_time)

    if base.startswith("p95-bucket-sjf"):
        return _pop_p95_bucket_sjf(w, profile_data, current_time)

    if base.startswith("p95-first"):
        return _pop_p95_first(w, profile_data, current_time)

    return w["queue"].pop(0)


def run_simulation(
    requests: list[dict],
    workers: list[dict],
    profile_data: dict,
    rps: float,
    t_end: float,
    algorithm: str,
    instance_scheduler_policy: str = "fcfs",
    chunk_preempt_budget_steps: int = 4,
    chunk_preempt_small_request_threshold_ms: float | None = None,
    chunk_preempt_image_budget_steps: int | None = None,
    chunk_preempt_video_budget_steps: int | None = None,
    use_heuristic_cost: bool = False,
) -> dict:
    """单次模拟，返回与 plot_results 对齐的指标 + algorithm。

    instance_scheduler_policy: 实例内调度策略，与 benchmark --instance-scheduler-policy 对齐。
      支持: fcfs, sjf, sjf_chunk_preempt (sjf_preempt), sjf_aging, sjf_aging_guarded,
            size_bucket_sjf_aging, p95-first, p95-bucket-sjf, p95-bucket-sjf-normalized
      后缀 _no_preempt 可关闭 chunk 抢占（如 sjf_aging_no_preempt）。
    chunk_preempt_*: 对齐 diffusion_engine / stage1 的 chunk 预算及 small_request 阈值。
      仅对 _policy_uses_chunk_preempt 返回 True 的策略生效。
    use_heuristic_cost: 调度排序使用面积 heuristic 而非 profile 成本估算，
      对齐 real deployment 未配置 --diffusion-runtime-profile 时的行为。
    """
    if algorithm not in DISPATCH:
        raise ValueError(f"不支持的算法: {algorithm}，可选: {list(DISPATCH.keys())}")
    dispatch_fn = DISPATCH[algorithm]
    state = {"profile_data": profile_data}
    _base_pol = instance_scheduler_policy.removesuffix("_no_preempt")
    is_chunk_preempt = _policy_uses_chunk_preempt(instance_scheduler_policy)

    # 对 sjf_chunk_preempt_N / sjf_preempt_N，从策略名解析 chunk budget
    chunk_budget = chunk_preempt_budget_steps
    if instance_scheduler_policy.startswith(("sjf_chunk_preempt_", "sjf_preempt_")):
        try:
            chunk_budget = int(instance_scheduler_policy.rsplit("_", 1)[1])
        except ValueError:
            pass

    # 复制 worker 状态：config_id, queue, next_time
    ws = []
    for w in workers:
        cid = worker_config_id(w)
        wr = {
            "config_id": cid,
            "instance_type": w.get("instance_type", cid),
            "queue": [],
            "next_time": INF,
            "_use_heuristic_cost": use_heuristic_cost,
        }
        if algorithm in ("short_queue_runtime", "short_queue_runtime_max_class_balanced"):
            wr["_queued_work"] = 0.0
        if algorithm == "short_queue_runtime_max_class_balanced":
            wr["_max_class_count"] = 0
        _base_pol = instance_scheduler_policy.removesuffix("_no_preempt")
        if _base_pol.startswith("p95"):
            wr["_p95_slowdown_history"] = []
            wr["_p95_service_rate_ms_per_wu"] = None
            wr["_p95_latency_history_ms"] = []
        ws.append(wr)
    n_workers = len(ws)
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

        # 请求到达
        if t == scheduler_next and req_index[0] < len(requests):
            req = next_request()
            if req is not None:
                state["current_time"] = t
                wi = dispatch_fn(req, ws, state)
                req["assigned_worker_index"] = wi
                req["assigned_worker_config"] = ws[wi]["config_id"]
                if algorithm in ("short_queue_runtime", "short_queue_runtime_max_class_balanced"):
                    st = lookup(profile_data, req, ws[wi])
                    req["_svc"] = st
                    ws[wi]["_queued_work"] += st
                if algorithm == "short_queue_runtime_max_class_balanced":
                    max_st = _max_service_time(profile_data)
                    if st >= max_st * 0.99:
                        ws[wi]["_max_class_count"] += 1
                ws[wi]["queue"].append(req)
            scheduler_next = scheduler_next_time()

        # Worker 完成
        for w in ws:
            if w["next_time"] == t:
                rec = w.get("current_request")
                w["next_time"] = INF
                w["current_request"] = None
                if rec is not None:
                    total_steps = max(int(rec.get("steps", 1) or 1), 1)
                    executed_before = int(rec.get("_executed_steps", 0) or 0)
                    chunk_done = int(rec.get("_chunk_steps", 0) or 0)
                    executed_after = executed_before + chunk_done
                    chunk_duration = rec.get("_last_chunk_duration_s", 0.0)

                    if is_chunk_preempt and executed_after < total_steps:
                        # Chunk 完成但请求未完成：记录 p95 样本后重新入队
                        if _base_pol.startswith("p95") and chunk_done > 0:
                            _p95_record_chunk_sample(w, chunk_duration, rec)
                        rec["_executed_steps"] = executed_after
                        rem = _remaining_latency_s(rec, profile_data, w)
                        if algorithm in ("short_queue_runtime", "short_queue_runtime_max_class_balanced"):
                            w["_queued_work"] += rem
                        w["queue"].append(rec)
                    else:
                        # 请求完成
                        if is_chunk_preempt:
                            rec["_executed_steps"] = executed_after
                        if _base_pol.startswith("p95") and chunk_done > 0:
                            _p95_record_chunk_sample(w, chunk_duration, rec)
                        if algorithm == "short_queue_runtime_max_class_balanced":
                            max_st = _max_service_time(profile_data)
                            if lookup(profile_data, rec, w) >= max_st * 0.99:
                                w["_max_class_count"] -= 1
                        rec["finish_time"] = t
                        rec["latency"] = t - rec["arrival_time"]
                        if _base_pol.startswith("p95"):
                            svc = rec.get("service_time") or sum(float(x) for x in rec.get("chunk_service_times", []))
                            if rec["latency"] > 0 and svc > 0:
                                _p95_record_completion(w, rec["latency"], svc)
                            _p95_record_latency_ms(w, rec["latency"] * 1000.0)
                        completed.append(rec)

        # 立即派工
        for w in ws:
            if w["next_time"] == INF and w["queue"]:
                req = _pop_next_from_queue(w, profile_data, instance_scheduler_policy, current_time=t)
                if algorithm in ("short_queue_runtime", "short_queue_runtime_max_class_balanced"):
                    sub = _remaining_latency_s(req, profile_data, w) if is_chunk_preempt else req.get("_svc", 0.0)
                    w["_queued_work"] -= sub
                if "first_start_time" not in req:
                    req["first_start_time"] = t
                req["start_time"] = req["first_start_time"]
                req.setdefault("start_times", []).append(t)
                if is_chunk_preempt:
                    per_req_budget = _sjf_preempt_budget_steps_for_request(
                        req, chunk_budget,
                        chunk_preempt_image_budget_steps,
                        chunk_preempt_video_budget_steps,
                    )
                    chunk_dur, chunk_steps = _chunk_duration_s(
                        req, profile_data, w, per_req_budget,
                        small_request_threshold_ms=chunk_preempt_small_request_threshold_ms,
                    )
                    req["_chunk_steps"] = chunk_steps
                    req["_last_chunk_duration_s"] = chunk_dur
                    st = chunk_dur
                else:
                    st = lookup(profile_data, req, w)
                req.setdefault("chunk_service_times", []).append(st)
                w["next_time"] = t + st
                w["current_request"] = req

    # 未完成的请求：仍在队列中的不记入 completed，无 failed 计数（按 README 可扩展）
    n_ok = len(completed)
    for r in completed:
        if "latency" not in r and "finish_time" in r and "arrival_time" in r:
            r["latency"] = r["finish_time"] - r["arrival_time"]
        chunk_times = r.get("chunk_service_times", [])
        if chunk_times:
            r["service_time"] = sum(float(x) for x in chunk_times)
        elif "finish_time" in r and "start_time" in r:
            r["service_time"] = r["finish_time"] - r["start_time"]
        if "latency" in r and "service_time" in r:
            r["waiting_time"] = max(r["latency"] - r["service_time"], 0.0)

    # 每个请求的明细：分配给的 worker、发起时间、开始执行时间、完成时间等
    def _round4(x):
        return round(x, 4) if isinstance(x, (int, float)) else x

    per_request = []
    for r in completed:
        start_times = r.get("start_times", [])
        chunk_service_times = r.get("chunk_service_times", [])
        per_request.append({
            "request_id": r.get("request_id"),
            "assigned_worker_index": r.get("assigned_worker_index"),
            "assigned_worker_config": r.get("assigned_worker_config"),
            "arrival_time": _round4(r["arrival_time"]),
            "start_time": _round4(r["start_time"]),
            "first_start_time": _round4(r.get("first_start_time", r.get("start_time"))),
            "start_times": [_round4(x) for x in start_times],
            "finish_time": _round4(r["finish_time"]),
            "latency": _round4(r["latency"]),
            "waiting_time": _round4(r["waiting_time"]),
            "service_time": _round4(r["service_time"]),
            "chunk_service_times": [_round4(x) for x in chunk_service_times],
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
    sched = cfg.get("scheduler", {})
    algorithms_to_run = sched.get("algorithms_to_run", [sched.get("algorithm", "round_robin")])
    if isinstance(algorithms_to_run, str):
        algorithms_to_run = [algorithms_to_run]
    policies_raw = sched.get("instance_scheduler_policies") or sched.get("instance_scheduler_policy", "fcfs")
    policies_flat = (
        [str(p).strip().lower() for p in policies_raw]
        if isinstance(policies_raw, (list, tuple))
        else [str(policies_raw).strip().lower()]
    )
    # sjf_aging / size_bucket_sjf_aging 可展开为多个 factor；默认 1.0 与 stage1_scheduler._SJF_AGING_DEFAULT_FACTOR 对齐
    sjf_aging_factors = sched.get("sjf_aging_factors")
    if sjf_aging_factors is None:
        sjf_aging_factors = [_SJF_AGING_DEFAULT_FACTOR]
    if not isinstance(sjf_aging_factors, (list, tuple)):
        sjf_aging_factors = [float(sjf_aging_factors)]
    # sjf_chunk_preempt / sjf_preempt 可展开为多个 chunk_budget
    chunk_preempt_budgets = sched.get("chunk_preempt_chunk_budgets") or sched.get("sjf_preempt_chunk_budgets")
    if chunk_preempt_budgets is None:
        chunk_preempt_budgets = [4]
    if not isinstance(chunk_preempt_budgets, (list, tuple)):
        chunk_preempt_budgets = [int(chunk_preempt_budgets)]
    instance_policies = []
    for p in policies_flat:
        if p == "sjf_aging":
            instance_policies.extend(f"sjf_aging_{f}" for f in sjf_aging_factors)
        elif p == "size_bucket_sjf_aging":
            instance_policies.extend(f"size_bucket_sjf_aging_{f}" for f in sjf_aging_factors)
        elif p in ("sjf_chunk_preempt", "sjf_preempt"):
            instance_policies.extend(f"sjf_chunk_preempt_{b}" for b in chunk_preempt_budgets)
        else:
            instance_policies.append(p)
    # 笛卡尔乘积：(algo, policy) 每个组合视为一个算法，输出 algorithm="{algo}_{policy}"
    algorithm_combos = [(a, p) for a in algorithms_to_run for p in instance_policies]
    combo_names = [f"{a}_{p}" for a, p in algorithm_combos]

    output_dir = Path(sim["output_dir"])
    output_merged = sim.get("output_merged", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _sched_optional_int(key: str) -> int | None:
        v = sched.get(key)
        if v in (None, ""):
            return None
        return int(v)

    img_chunk_bs = _sched_optional_int("chunk_preempt_image_budget_steps") or _sched_optional_int("sjf_preempt_image_chunk_budget_steps")
    vid_chunk_bs = _sched_optional_int("chunk_preempt_video_budget_steps") or _sched_optional_int("sjf_preempt_video_chunk_budget_steps")
    default_chunk_budget = int(sched.get("chunk_preempt_budget_steps", 4) or 4)

    all_runs = []
    for algo, instance_policy in algorithm_combos:
        combo_name = f"{algo}_{instance_policy}"
        runs_for_combo = []
        for rps in rps_list:
            requests = build_requests(cfg, rps, t_end)
            assign_slo(requests, profile_data, workers, slo_scale)
            try:
                thresh = sched.get("chunk_preempt_small_request_threshold_ms") or sched.get("sjf_preempt_small_request_threshold_ms")
                thresh_f = float(thresh) if thresh not in (None, "") else None
                out = run_simulation(
                    requests, workers, profile_data, rps, t_end, algo,
                    instance_scheduler_policy=instance_policy,
                    chunk_preempt_budget_steps=default_chunk_budget,
                    chunk_preempt_small_request_threshold_ms=thresh_f,
                    chunk_preempt_image_budget_steps=img_chunk_bs,
                    chunk_preempt_video_budget_steps=vid_chunk_bs,
                )
                out["algorithm"] = combo_name  # 覆盖为组合名，画图时每条线一个组合
                runs_for_combo.append(out)
                all_runs.append(out)
            except (KeyError, ValueError) as e:
                print(f"算法 {combo_name} rps={rps} 运行失败: {e}", file=sys.stderr)
                sys.exit(1)
        if not output_merged:
            runs_stats = [{k: v for k, v in run.items() if k != "requests"} for run in runs_for_combo]
            out_path = output_dir / f"{combo_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(runs_stats, f, indent=2)
            req_path = output_dir / f"{combo_name}_requests.csv"
            _write_requests_csv(req_path, runs_for_combo)
            print(f"已写入 {combo_name}: 统计 {out_path}，请求明细 {req_path} ({len(runs_for_combo)} 个 rps 点)")

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
        append_files = plot_cfg.get("append_runs_files") or []
        if isinstance(append_files, (str, Path)):
            append_files = [append_files]
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

            # 追加外部 runs（例如 real 指标点），与 plot_runs 合并后一起画图
            base = config_path.parent
            for ap in append_files:
                ap_path = Path(ap)
                if not ap_path.is_absolute():
                    ap_path = base / ap_path
                if not ap_path.exists():
                    raise FileNotFoundError(f"append_runs_files 不存在: {ap_path}")
                with open(ap_path, encoding="utf-8") as f:
                    extra = json.load(f)
                if isinstance(extra, dict) and "runs" in extra:
                    extra = extra["runs"]
                if not isinstance(extra, list):
                    raise ValueError(f"append_runs_files 内容必须是 list[dict]（或 {{'runs': [...]}}），得到: {type(extra)}: {ap_path}")
                for i, run in enumerate(extra):
                    if not isinstance(run, dict):
                        raise ValueError(f"append run[{i}] 必须为 dict，得到 {type(run)}: {ap_path}")
                    if "algorithm" not in run or "rps" not in run:
                        raise ValueError(f"append run[{i}] 缺少 algorithm/rps: {ap_path}")
                    for m in plot_metrics:
                        if m not in run:
                            raise ValueError(f"append run[{i}] 缺少绘图指标 {m!r}: {ap_path}")
                    plot_runs.append(run)
            with open(merged_for_plot, "w", encoding="utf-8") as f:
                json.dump(plot_runs, f, indent=2)
            saved_list = []
            for metric in plot_metrics:
                out_path = plot_output_dir / f"{plot_output_prefix}_{metric}.png"
                n_workers = len(workers)
                t_end_str = str(int(t_end)) if t_end == int(t_end) else str(t_end)
                saved = plot_results_fn(
                    merged_for_plot,
                    output=out_path,
                    metrics=[metric],
                    title=f"Simulation (t_end={t_end_str}s, workers={n_workers})",
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

    print("完成。画图示例: python diffusion_bench/plot_results.py --input-dir", output_dir, "--algorithms", " ".join(combo_names), "-o figs/compare --split-by-type")


if __name__ == "__main__":
    main()
