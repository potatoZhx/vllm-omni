#!/usr/bin/env python3
# 对 qwen_profile_*.csv 做聚合：每 5 行（run1~5）聚合成 1 行。
# 规则：丢弃 run1，对 run2、3、4、5 的 request_time_s 求平均作为新的 request_time_s。
# 输出列：size, steps, config_id(翻译为 spX_cfgY_tpZ), request_time_s
# 聚合前检查同一组内 size、steps、config_id 一致，否则报错退出。

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_config_mapping(mapping_path: Path) -> dict[int, str]:
    """加载 config_id -> config_label 映射（来自 config_id_mapping.csv）。"""
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if "config_id" in row and "config_label" in row:
                mapping[int(row["config_id"].strip())] = row["config_label"].strip()
    return mapping


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python3 aggregate_profile_csv.py <输入.csv> [config_id_mapping.csv]", file=sys.stderr)
        print("  输出: profile_qwen.csv（与输入同目录）", file=sys.stderr)
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    if not input_csv.exists():
        print(f"错误: 输入文件不存在: {input_csv}", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    mapping_file = Path(sys.argv[2]) if len(sys.argv) > 2 else script_dir / "config_id_mapping.csv"
    if not mapping_file.exists():
        print(f"错误: 映射文件不存在: {mapping_file}", file=sys.stderr)
        sys.exit(1)

    config_label = load_config_mapping(mapping_file)
    if not config_label:
        print("错误: 映射文件为空或格式错误", file=sys.stderr)
        sys.exit(1)

    out_csv = input_csv.parent / "profile_qwen.csv"

    rows: list[dict] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    expected_cols = {"size", "steps", "config_id", "run", "request_time_s"}
    if not expected_cols.issubset(set(rows[0].keys()) if rows else set()):
        print("错误: 输入 CSV 需包含列: size, steps, config_id, run, request_time_s", file=sys.stderr)
        sys.exit(1)

    out_rows: list[dict] = []
    n = len(rows)
    if n % 5 != 0:
        print(f"错误: 行数 {n} 不是 5 的倍数，无法每 5 行聚合", file=sys.stderr)
        sys.exit(1)

    for i in range(0, n, 5):
        chunk = rows[i : i + 5]
        sizes = [r["size"].strip() for r in chunk]
        steps_list = [r["steps"].strip() for r in chunk]
        config_ids = [r["config_id"].strip() for r in chunk]
        runs = [r["run"].strip() for r in chunk]
        times = [float(r["request_time_s"].strip()) for r in chunk]

        if len(set(sizes)) != 1 or len(set(steps_list)) != 1 or len(set(config_ids)) != 1:
            print(
                f"错误: 第 {i+2}–{i+6} 行（0-based 块 {i//5}）中 size/steps/config_id 不一致: "
                f"size={sizes}, steps={steps_list}, config_id={config_ids}",
                file=sys.stderr,
            )
            sys.exit(1)

        # run 应为 1,2,3,4,5
        if set(runs) != {"1", "2", "3", "4", "5"}:
            print(
                f"错误: 第 {i+2}–{i+6} 行 run 应为 1,2,3,4,5，实际为: {runs}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 丢弃 run1，对 run2,3,4,5 的 request_time_s 求平均
        run2_5_times = [t for r, t in zip(runs, times) if r != "1"]
        avg_time = sum(run2_5_times) / len(run2_5_times)

        cid = int(config_ids[0])
        if cid not in config_label:
            print(f"错误: config_id={cid} 在映射表中不存在", file=sys.stderr)
            sys.exit(1)

        out_rows.append({
            "size": sizes[0],
            "steps": steps_list[0],
            "config_id": config_label[cid],
            "request_time_s": f"{avg_time:.4f}",
        })

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "steps", "config_id", "request_time_s"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"已写入 {len(out_rows)} 行 -> {out_csv}")


if __name__ == "__main__":
    main()
