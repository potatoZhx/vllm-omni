# `v16-base` Step-Level / Benchmark / Global Scheduler 合入 `v18-base` 简要工作进度

> 长版文档: `workplan/work_progress.md`
>
> 对齐方案: `workplan/v16_to_v18_step_level_minimal_landing_plan.md`
>
> 进度截点: `2026-04-08`

## 0. 当前状态

当前已经完成的主线工作可以概括为三件事:

- `v18-base` 上的 step-level diffusion 主闭环已经打通
- benchmark 与 global scheduler 的最小可用链路已经恢复
- 为了后续做真实策略对比，又补齐了观测日志和 benchmark 可复现性

当前系统已经能够支持:

- `step_level_request_scheduler + diffusion_enable_step_chunk=True`
- 实例内 `fcfs / sjf / sjf_aging / sjf_aging_guarded / sjf_aging_guarded_tail / p95-first`
- 最小可用的 global scheduler runtime
- benchmark 侧 `estimated_cost_s` 注入
- benchmark arrival trace 默认固定 seed

当前仍未覆盖:

- `batch_size > 1`
- `generate_batch()` / multi-prompt 主路径
- `p95-fusion` 等更复杂策略
- 完整 trace replay / trace 落盘

## 1. 时间线概览

| 时间 | 本轮改动了什么 | 主要修改内容 | 修改逻辑 |
| --- | --- | --- | --- |
| `2026-04-01` | Step-level 主闭环最小落地 | config、scheduler state、step-level backend、engine step loop、abort、executor stepwise transport | 先把 `scheduler -> engine -> executor/worker` 这条最小闭环打通，不回退到 `v16-base` 的一体化结构 |
| `2026-04-02` | benchmark 能力迁移 | warmup request profile、`estimated_cost_s` / `slo_ms` 注入、结果保存、Wan 单实例脚本 | 先补 benchmark 侧“可测”和“可给 scheduler 喂 cost”能力，为后续策略验证铺路 |
| `2026-04-02` | global scheduler 最小 runtime 恢复 | `config/state/router/server/process_controller`、3 个全局策略、worker metadata 透传、精简 orchestration | 只恢复 pure routing 版本，不让请求堆在 global scheduler 侧，避免过早引入新的状态复杂度 |
| `2026-04-06` | 实例内策略迁移与模板文档收尾 | `sjf/sjf_aging/sjf_aging_guarded/sjf_aging_guarded_tail/p95-first`、模板、README、中英文文档 | 先迁 waiting-queue selection 语义，再补文档和模板，让运行入口与实现对齐 |
| `2026-04-07` ~ `2026-04-08` | 观测性与 benchmark 可复现性修复 | `StepSchedule/StepComplete`、executor lifecycle 日志、默认不再 `max_workers=1`、`--arrival-seed` 默认固定 | 先把“看不见 / 复现不了”的问题解决，再做真实日志分析和策略公平对比 |

## 2. 每轮工作内容

### 2.1 `2026-04-01` Step-Level 主闭环最小落地

本轮改动了什么:

- 配置入口支持 `step_level_request_scheduler`
- scheduler 拥有最小执行态
- engine 改为中心化 step loop
- executor/worker 打通 `execute_stepwise()`
- `abort()` 收敛路径补齐

主要修改内容:

- `vllm_omni/diffusion/data.py`
  - 增加 step-level backend 配置和组合校验
- `vllm_omni/diffusion/sched/interface.py`
  - 增加 `DiffusionExecutionState`
- `vllm_omni/diffusion/sched/base_scheduler.py`
  - 增加 execution state、request id 映射、abort pending
- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 新增 step-level backend
- `vllm_omni/diffusion/diffusion_engine.py`
  - 增加 scheduler thread、step loop、waiter/output 收敛
- `vllm_omni/diffusion/executor/multiproc_executor.py`
  - 新增 stepwise transport

修改逻辑:

- 不把旧版 `v16-base` 的 scheduler 整体搬回来
- 保持 `scheduler` 管状态、`engine` 管循环、`executor/worker` 管执行
- 先只保证单请求、非 batch 的 step-level 主路径成立

对应关键提交:

- `21a25d29` 配置与 contract 脚手架
- `8646ea21` scheduler kernel + policy 抽象 + executor stepwise transport
- `366fdb31` engine step loop + abort 收敛
- `1abe14df` async abort 覆盖修复
- `14427b41` worker import 初始化修复
- `47409a96` engine rpc/state 锁拆分

### 2.2 `2026-04-02` Benchmark 能力迁移

本轮改动了什么:

- benchmark 侧恢复 warmup request profile
- 恢复 `estimated_cost_s` / `slo_ms` 注入
- 恢复结果保存能力
- 补齐单实例 `Wan2.2` / `Qwen` 模板与脚本基础

主要修改内容:

- `benchmarks/diffusion/backends.py`
  - `RequestFuncInput` 增加 `estimated_cost_s`
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
  - 增加 `--warmup-request-config`
  - warmup 推导 `estimated_cost_s` / `slo_ms`
  - 默认注入 `estimated_cost_s`
  - 增加 `--save-output-dir`
- `benchmarks/diffusion/scripts/run_wan_sp4_cfg2_hsdp_rps_bench.sh`
  - 恢复单实例 RPS sweep 脚本

修改逻辑:

- step-level / scheduler 要想比较策略，benchmark 端必须先能稳定提供 cost 信息
- 不回退当前 `v18-base` 的视频异步 job 语义，只补对实验有用的增强

### 2.3 `2026-04-02` Global Scheduler 最小 runtime 恢复

本轮改动了什么:

- 恢复最小 global scheduler runtime
- 恢复 3 个目标全局策略
- 恢复精简的 orchestration 与 benchmark 脚本
- worker 侧开始接受并透传 `slo_ms` / `estimated_cost_s`

主要修改内容:

- `vllm_omni/global_scheduler/*`
  - 恢复 `config.py`、`state.py`、`router.py`、`server.py`、`lifecycle.py` 等
- `vllm_omni/global_scheduler/policies/*`
  - 恢复 `min_queue_length`
  - 恢复 `round_robin`
  - 恢复 `short_queue_runtime`
- OpenAI serving 入口
  - 图像 / 视频 / chat 路径开始透传 scheduler metadata
- `benchmarks/diffusion/scripts/global_instance_scheduler_v2/*`
  - 恢复最小 orchestration

修改逻辑:

- 只恢复 pure routing，不做 scheduler-side waiting queue
- 避免 global scheduler 本身变成新的排队瓶颈
- worker 内部执行顺序仍交给本地 scheduler 决定

### 2.4 `2026-04-06` 实例内策略迁移与文档模板收尾

本轮改动了什么:

- 从 `v16-base` 迁入第一批 waiting-queue 策略
- 补齐 benchmark / orchestration 模板
- 补齐中英文 README

主要修改内容:

- `vllm_omni/diffusion/sched/policy.py`
  - 迁入 `sjf`
  - 迁入 `sjf_aging`
  - 迁入 `sjf_aging_guarded`
  - 迁入 `sjf_aging_guarded_tail`
  - 迁入 `p95-first`
- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 增加 policy lifecycle hook
- `benchmarks/diffusion/scripts/global_instance_scheduler_v2/*.yaml`
  - 单实例模板改为显式 warmup profile
- `README.md` / `README_zh.md`
  - 文档与模板对齐

修改逻辑:

- 只迁 waiting-queue selection 语义，不迁旧版全部 chunk-budget 行为
- 先让实例内策略在新架构上“能跑、能测、能说明白”

### 2.5 `2026-04-07` ~ `2026-04-08` 观测性与 benchmark 可复现性修复

本轮改动了什么:

- 补齐 step-level 调度日志
- 补齐 executor 入口日志
- 修复默认 `max_workers=1` 导致的实例内请求人工串行
- 修复 benchmark arrival trace 不可复现

主要修改内容:

- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 新增 `[StepSchedule]`
  - 新增 `[StepComplete]`
- `vllm_omni/entrypoints/async_omni_diffusion.py`
  - executor worker 数改为可配置
  - 默认不再固定为 `1`
  - 新增：
    - `[AsyncDiffusionExecutorEnqueue]`
    - `[AsyncDiffusionExecutorStart]`
    - `[AsyncDiffusionExecutorDone]`
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
  - 新增 `DEFAULT_ARRIVAL_SEED = 42`
  - 新增 `--arrival-seed`
  - `iter_requests()` 改为独立 `random.Random(arrival_seed)`

修改逻辑:

- 先解决“日志里看不到 step-level 真正发生了什么”的问题
- 再解决“worker 入口被单线程 executor 串住”的问题
- 最后解决“同 RPS rerun arrival trace 不同，导致实验不公平”的问题

这一轮定位出的结论:

- old benchmark 的主要问题不是请求 mix 飘了
- 请求 profile 基本是固定的
- 真正漂的是 arrival trace
- 在旧 `max_workers=1` 条件下，arrival burstiness 会被放大成很明显的 latency 漂移

## 3. 当前代码改动的核心逻辑

如果只看主线逻辑，整个工作可以归纳成 4 条：

1. 先把 step-level 主闭环做成 `v18-base` 风格的最小实现。
   也就是保留当前分层，不回退到旧版一体化 scheduler。

2. 再把 benchmark 和 global scheduler 恢复到“能稳定喂请求、能透传 cost、能跑最小实验”的状态。
   这一步的重点不是功能做全，而是保证数据链路打通。

3. 然后把第一批实例内策略迁上来。
   先迁 waiting-queue selection 语义，复杂策略细节后补。

4. 最后补观测性和可复现性。
   因为没有逐 step 日志、executor 入口被串行、arrival trace 不固定时，实验现象很难解释，也很难公平比较。

## 4. 已完成验证

已完成的定向验证包括:

- step-level scheduler 定向 pytest
- async diffusion executor 定向 pytest
- benchmark arrival seed 定向 pytest
- global scheduler 定向 pytest
- `vllm serve` 真实 smoke
- benchmark baseline / experiment 10 请求性能对比

最近一轮新增验证:

- `tests/diffusion/test_diffusion_scheduler.py`
  - `25 passed`
- `tests/entrypoints/test_async_omni_diffusion.py`
  - `2 passed`
- `tests/benchmarks/test_diffusion_benchmark_serving.py`
  - `2 passed`

## 5. 当前结论

- 主闭环已经不是“设计完成”，而是已经能运行、能测试、能从日志里分析行为
- global scheduler 与 benchmark 的最小实验链路已经恢复
- 实例内策略第一批已经迁入
- benchmark 的 arrival trace 默认已经可复现
- 接下来的重点不再是补最小骨架，而是：
  - 更复杂策略迁移
  - 更大规模真实 benchmark / soak
  - 更严格的 trace replay 与实验公平性控制
