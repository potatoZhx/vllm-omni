#!/usr/bin/env python3
"""
t_end=18000s 仿真结果严格分析：
1. sjf_aging_factors 最优选择
2. max_class_balanced 可行性判断
"""

import json
from pathlib import Path

OUTPUT_DIR = "output/newest_profile_A100_n8_short_queue_runtime_short_queue_runtime_max_class_balanced_fcfs_sjf_sjf_aging"
RPS_VALS = [0.2, 0.4, 0.6, 0.8, 1.0]

BASE_ALGOS = ["short_queue_runtime"]
BAL_ALGOS = ["short_queue_runtime_max_class_balanced"]
POLICIES = ["fcfs", "sjf", "sjf_aging_0.15", "sjf_aging_0.2", "sjf_aging_0.25"]


def load(name: str, dir_path: Path) -> list:
    with open(dir_path / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    base = Path(__file__).parent
    dir_path = base / OUTPUT_DIR
    if not dir_path.exists():
        dir_path = base.parent / OUTPUT_DIR

    print("=" * 90)
    print("t_end=18000s 仿真严格分析")
    print("=" * 90)
    print("输出目录:", dir_path)
    print()

    # ---- 1. 全表：rps=1.0 高负载下各组合 P95 / Mean / Throughput ----
    print("\n### 1. 高负载 rps=1.0 全组合指标 ###\n")
    rows = []
    for algo in BASE_ALGOS + BAL_ALGOS:
        for pol in POLICIES:
            name = f"{algo}_{pol}"
            try:
                data = load(name, dir_path)
                r = data[4]  # rps=1.0
                rows.append({
                    "algo": algo,
                    "policy": pol,
                    "p95": r["latency_p95"],
                    "mean": r["latency_mean"],
                    "qps": r["throughput_qps"],
                    "name": name,
                })
            except Exception as e:
                print(f"Skip {name}: {e}")

    # 表头
    fmt = "{:45} {:>10} {:>10} {:>10}"
    print(fmt.format("Algorithm_Policy", "P95(s)", "Mean(s)", "QPS"))
    print("-" * 80)
    for r in sorted(rows, key=lambda x: (x["algo"], x["policy"])):
        print(fmt.format(r["name"], f"{r['p95']:.1f}", f"{r['mean']:.1f}", f"{r['qps']:.4f}"))

    # ---- 2. sjf_aging_factors 最优 ----
    print("\n### 2. sjf_aging_factors 最优分析 (rps=1.0, 以 P95 为主) ###\n")
    aging_policies = [p for p in POLICIES if p.startswith("sjf_aging")]
    for algo in BASE_ALGOS + BAL_ALGOS:
        best_p95 = None
        best_pol = None
        for pol in aging_policies:
            name = f"{algo}_{pol}"
            try:
                r = load(name, dir_path)[4]
                if best_p95 is None or r["latency_p95"] < best_p95:
                    best_p95 = r["latency_p95"]
                    best_pol = pol
            except Exception:
                pass
        if best_pol:
            print(f"{algo}: 最优 policy={best_pol}, P95={best_p95:.1f}s")

    # ---- 3. max_class_balanced 可行性：base vs balanced 配对对比 ----
    print("\n### 3. max_class_balanced 可行性 (base vs balanced 配对) ###\n")
    print(fmt.format("Policy", "Base_P95", "Bal_P95", "P95_Delta%"))
    print("-" * 80)
    for pol in POLICIES:
        base_name = f"short_queue_runtime_{pol}"
        bal_name = f"short_queue_runtime_max_class_balanced_{pol}"
        try:
            b = load(base_name, dir_path)[4]
            a = load(bal_name, dir_path)[4]
            delta = (a["latency_p95"] - b["latency_p95"]) / b["latency_p95"] * 100 if b["latency_p95"] else 0
            verdict = "可行" if delta <= 0 else "不可行"
            print(fmt.format(pol, f"{b['latency_p95']:.1f}", f"{a['latency_p95']:.1f}", f"{delta:+.1f}%") + f"  [{verdict}]")
        except Exception as e:
            print(f"{pol}: 错误 {e}")

    # ---- 4. 全 RPS 下 max_class_balanced 收益/损失统计 ----
    print("\n### 4. 全 RPS max_class_balanced P95 变化汇总 ###\n")
    for pol in POLICIES:
        base_name = f"short_queue_runtime_{pol}"
        bal_name = f"short_queue_runtime_max_class_balanced_{pol}"
        try:
            b_data = load(base_name, dir_path)
            a_data = load(bal_name, dir_path)
            wins = sum(1 for i in range(5) if a_data[i]["latency_p95"] < b_data[i]["latency_p95"])
            deltas = [(a_data[i]["latency_p95"] - b_data[i]["latency_p95"]) / b_data[i]["latency_p95"] * 100
                        for i in range(5) if b_data[i]["latency_p95"] > 0]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            print(f"{pol:25} 胜出次数: {wins}/5, 平均P95变化: {avg_delta:+.1f}%")
        except Exception as e:
            print(f"{pol}: {e}")

    # ---- 5. 全 RPS 详细表（P95）----
    print("\n### 5. 全 RPS P95 详细 (s) ###\n")
    print(f"{'RPS':>5}", end=" ")
    for pol in POLICIES[:3]:  # fcfs, sjf, sjf_aging_0.15
        print(f"{'base':>8} {'bal':>8}", end=" ")
    print()
    print("-" * 60)
    for ri in range(5):
        rps = RPS_VALS[ri]
        print(f"{rps:5.2f}", end=" ")
        for pol in POLICIES[:3]:
            try:
                b = load(f"short_queue_runtime_{pol}", dir_path)[ri]["latency_p95"]
                a = load(f"short_queue_runtime_max_class_balanced_{pol}", dir_path)[ri]["latency_p95"]
                print(f"{b:8.1f} {a:8.1f}", end=" ")
            except Exception:
                print("  --    -- ", end=" ")
        print()

    # ---- 6. 最终推荐 ----
    print("\n### 6. 最终推荐 ###\n")
    # 最优 sjf_aging
    best_aging = None
    best_p95 = 1e9
    for pol in aging_policies:
        for algo in [BASE_ALGOS[0], BAL_ALGOS[0]]:
            try:
                r = load(f"{algo}_{pol}", dir_path)[4]
                if r["latency_p95"] < best_p95:
                    best_p95 = r["latency_p95"]
                    best_aging = (algo, pol)
            except Exception:
                pass
    print(f"1) 最优 sjf_aging 组合: {best_aging[0]}_{best_aging[1]}, P95={best_p95:.1f}s")

    # max_class_balanced 是否整体推荐
    bal_better = 0
    bal_worse = 0
    for pol in POLICIES:
        try:
            b = load(f"short_queue_runtime_{pol}", dir_path)[4]
            a = load(f"short_queue_runtime_max_class_balanced_{pol}", dir_path)[4]
            if a["latency_p95"] < b["latency_p95"]:
                bal_better += 1
            else:
                bal_worse += 1
        except Exception:
            pass
    print(f"2) max_class_balanced 可行性: 5 policies 中 {bal_better} 个改善、{bal_worse} 个变差")
    if bal_better >= 4:
        print("   -> 推荐：可默认启用 max_class_balanced")
    elif bal_worse >= 4:
        print("   -> 不推荐：多数 policy 下 P95 变差")
    else:
        print("   -> 建议：按 instance policy 选择性启用（fcfs/sjf_aging 可试，sjf 慎用）")


if __name__ == "__main__":
    main()
