# `v16-base` Step-Level 主闭环 + Benchmark 迁移合入 `v18-base` 的当前工作进度

> 对齐文档: `workplan/v16_to_v18_step_level_minimal_landing_plan.md`
>
> 进度截点: `2026-04-08`
>
> 当前代码基线: `v18-base` `e6e4e4c8` + global scheduler / benchmark doc-template working tree

## 0. 结论先行

当前最小可落地版本已经在 `v18-base` 上完成主闭环，并完成了代码级回归、真实 `vllm serve` 功能 smoke、以及基准组 / 实验组的 10 条请求性能一致性验证。

截至 `2026-04-08`，在 `2026-04-02` 已完成的 step-level + global scheduler 最小落地基础上，又补齐了 benchmark/orchestration 模板与中英文文档收尾、step-level / executor 逐步观测日志、以及 benchmark arrival trace 默认固定 seed，包括 `warmup_request_config` 接线、`Wan2.2` 单实例模板、benchmark 中文 README、global scheduler 模块级详细说明文档、`StepSchedule/StepComplete` 日志、executor enqueue/start/done 日志、以及 `--arrival-seed` 默认固定为 `42` 的可复现实验入口。

本轮实际落地与原方案保持一致的部分:

- 继续沿用 `#1625` 已确立的 `scheduler -> engine -> executor/worker/runner` 分层
- 没有把 `v16-base` 的历史一体化 scheduler 结构直接搬回 `v18-base`
- 已落地 `StepLevelRequestScheduler + policy 抽象 + engine 中心化 step loop + abort 闭环`
- 默认稳定路径仍保持 `request_scheduler + step_chunk=False`

本轮已完成的 MVP 边界:

- 支持 `step_level_request_scheduler + diffusion_enable_step_chunk=True`
- 支持 step 边界 unfinished continuation / requeue
- 支持 `abort()`
- 支持 `instance_scheduler_policy=fcfs`
- 已迁移 `sjf`
- 已迁移 `sjf_aging`
- 已迁移 `sjf_aging_guarded`
- 已迁移 `sjf_aging_guarded_tail`
- 已迁移 `p95-first`
- 只保证单请求、非 batch 的 step-level 主路径
- 已验证修改后代码与 `v18-base` 原生基准代码性能一致

本轮未进入实现范围的项:

- 独立 `RuntimeProfileEstimator service`
- `p95-fusion` / bucket / deadline / bypass-guard 等其余复杂策略
- `batch_size > 1`
- `generate_batch()` / multi-prompt / stage batch fast path

## 1. 设计基线

本轮实现严格按原方案的分层落地，而不是回退到 `v16-base` 的结构:

- `scheduler` 负责 request lifecycle、waiting/running 队列、policy 接入、abort pending、finished 状态
- `engine` 负责中心化调度循环、waiter 唤醒、terminal output 收敛、close/fatal error 收敛
- `executor` 负责 all-rank `execute_stepwise()` transport
- `worker/runner` 继续复用 `v18-base` 现有 stepwise contract 与 `state_cache`

本轮新增的现实约束:

- 为兼容 `v18-base` 当前代码和真实服务链路，额外修复了 `async abort` 覆盖问题、worker 包级导入初始化环、以及 engine 锁边界死锁风险
- 真实验证环境最终切换为全新 conda 环境 `/home/tianzhu/.conda/envs/vllm-omni-v18`，并统一使用 `PYTHONNOUSERSITE=1` 隔离用户站点包污染

## 2. 范围界定

### In Scope

| 项目 | 原方案预期 | 当前状态 | 证据 |
| --- | --- | --- | --- |
| `StepLevelRequestScheduler` | 落地 | 已完成 | `8646ea21` |
| scheduler-owned execution state | 落地 | 已完成 | `21a25d29` |
| engine 中心化 step loop | 落地 | 已完成 | `366fdb31`, `47409a96` |
| step 边界抢占 / continuation | 落地 | 已完成 | `8646ea21`, `366fdb31` |
| `abort()` 端到端打通 | 落地 | 已完成 | `366fdb31`, `1abe14df` |
| config / CLI / stage builder 参数透传 | 落地 | 已完成 | `21a25d29` |
| policy 抽象 | 落地 | 已完成 | `8646ea21` |
| 首版 `fcfs` policy | 落地 | 已完成 | `8646ea21` |
| `RuntimeProfileEstimator` 独立 service | 落地 | 未实现 | 本轮 defer |
| 真实 `vllm serve` 验证 | 验收要求 | 已完成 | benchmark / logs / metrics |

### Out of Scope

以下内容仍保持为未实现状态，与原方案一致:

- step 内中断
- `batch_size > 1` continuous batching
- `generate_batch()` / `StageDiffusionClient.add_batch_request_async()` / multi-prompt request
- step mode 下的 cache backend / KV transfer / LoRA 增强
- `p95-fusion` / `p95-first-deadline` / `p95-bucket-sjf` / `bypass_guard_sjf` 等后续策略
- global scheduler 合并
- 回退 `#1625` 的 executor / scheduler 分层

## 3. 最终架构

### 3.1 模块架构

本轮实际落地后的主链路为:

- `vllm_omni/diffusion/data.py`
  - 扩展 step-level backend 公开配置和组合校验
- `vllm_omni/engine/async_omni_engine.py`
  - 透传 diffusion scheduler backend / step chunk / policy 参数
- `vllm_omni/diffusion/sched/interface.py`
  - 新增 `DiffusionExecutionState` 和 `ExecutionOutput`
- `vllm_omni/diffusion/sched/base_scheduler.py`
  - 新增 execution state / abort pending / request id 映射维护
- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 新 backend 主体
- `vllm_omni/diffusion/sched/policy.py`
  - `RequestSelectionPolicy` 抽象
  - `FCFSSelectionPolicy`
  - `SJFSelectionPolicy`
  - `SJFAgingSelectionPolicy`
  - `SJFAgingGuardedSelectionPolicy`
  - `SJFAgingGuardedTailSelectionPolicy`
  - `P95FirstSelectionPolicy`
- `vllm_omni/diffusion/diffusion_engine.py`
  - step-level scheduler thread、waiter、abort/close/fatal error 收敛
- `vllm_omni/diffusion/executor/multiproc_executor.py`
  - all-rank `execute_stepwise()` transport

### 3.2 step-level 数据流

当前实现的数据流与原方案一致:

1. client 调用 `add_req_and_wait_for_response()`
2. scheduler 生成 `sched_req_id` 并入 waiting 队列
3. engine 后台 scheduler thread 调用 `schedule()`
4. executor 执行 `execute_stepwise()`
5. runner 返回 `RunnerOutput(req_id, step_index, finished, result)`
6. scheduler 根据 `RunnerOutput` 更新 execution state
7. unfinished 请求被重新入队，finished 请求由 engine 收敛成 terminal `DiffusionOutput`

### 3.3 状态所有权

当前状态所有权保持原方案约束:

- 生命周期真相源在 scheduler
- `executed_steps` 等执行态由 scheduler-owned `DiffusionExecutionState` 维护
- runner `state_cache` 仍归 runner 所有
- 最终 `DiffusionOutput` 由 engine 统一保存并唤醒 waiter
- `abort_pending` 由 engine 标记、scheduler 在 step 边界收敛

## 4. 最小支持矩阵

当前实际支持矩阵如下:

| backend | step_chunk | 当前结果 |
| --- | --- | --- |
| `request_scheduler` | `False` | 支持，保持默认稳定路径 |
| `step_level_request_scheduler` | `True` | 支持，本轮 MVP 主路径 |
| `request_scheduler` | `True` | 显式拒绝 |
| `step_level_request_scheduler` | `False` | 显式拒绝 |

额外限制与原方案一致:

- `instance_scheduler_policy` 当前接受：
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`
- `generate_batch()` / stage batch / multi-prompt request 直接报 `NotImplementedError`
- step-level 路径只保证单请求非 batch 主路径

## 5. 最终代码方案

### 5.1 配置与入口

已完成:

- `OmniDiffusionConfig` 新增 step-level backend 相关字段和组合校验
- `stage_config` / `AsyncOmniEngine` / CLI `serve` 透传上述字段
- 当 `diffusion_scheduler_backend == "step_level_request_scheduler"` 且 `diffusion_enable_step_chunk=True` 时，自动派生 `step_execution=True`

落地 commit:

- `21a25d29 Add step-level diffusion config scaffolding`

### 5.2 request / scheduler state

已完成:

- 新增 `DiffusionExecutionState`
- scheduler 基类增加:
  - `_execution_states`
  - `_request_id_to_sched_req_id`
  - `_abort_pending`
  - `get_execution_state()`
  - `get_sched_req_id()`
  - `mark_abort_pending()`
  - `is_abort_pending()`

说明:

- 本轮只补齐最小 scheduler-owned state，没有把更多镜像字段重新做回 request 真相源

### 5.3 新 backend

已完成:

- 新增 `StepLevelRequestScheduler`
- `RequestScheduler.update_from_output()` 收紧为只接受 `DiffusionOutput`
- step-level backend 收紧为只接受 `RunnerOutput`
- unfinished step 输出会回到 waiting 队列

落地 commit:

- `8646ea21 Add step-level diffusion scheduler kernel`

### 5.4 policy 层

已完成:

- 新增 `RequestSelectionPolicy`
- 新增 `FCFSSelectionPolicy`
- 新增 `SJFSelectionPolicy`
- 新增 `SJFAgingSelectionPolicy`
- 新增 `SJFAgingGuardedSelectionPolicy`
- 新增 `SJFAgingGuardedTailSelectionPolicy`
- 新增 `P95FirstSelectionPolicy`
- `build_request_selection_policy()` 已开放：
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`

当前状态:

- policy 框架已落地
- 第一批 `v16-base` waiting-queue 策略语义已迁入
- `sjf_aging_guard` 作为 alias 会归一化到 `sjf_aging_guarded`
- `sjf_aging_guarded_tail` 当前只迁入 waiting-queue tail sink 语义，尚未迁 chunk-budget override

### 5.5 runtime estimator

当前状态:

- 没有单独恢复 `v16-base` 的独立 `RuntimeProfileEstimator service`
- 但 step-level policy 已具备最小 runtime-aware 输入链路：
  - 请求 `estimated_cost_s`
  - `instance_runtime_profile_path`
  - 启发式 fallback

原因:

- 本轮目标是先把 `sjf / guarded / p95-first` 的 waiting-queue 语义迁进 `v18-base`
- 暂不把旧的一体化 runtime estimator service 结构整体搬回主干

### 5.6 engine 主闭环

已完成:

- engine 根据 config 自动选择 `RequestScheduler` 或 `StepLevelRequestScheduler`
- step-level 模式下启动后台 scheduler thread
- engine 内新增:
  - `_scheduler_cv`
  - `_request_events`
  - `_request_outputs`
  - `_fatal_error`
  - `_closed`
  - `_scheduler_thread`
- unfinished / finished / fatal error / close 都会在 engine 侧统一收敛

#### 5.6.1 engine 锁设计

最终实现采用两把锁:

- `_engine_lock`
  - 管 request 状态、scheduler 状态、events、outputs、close/fatal error
- `_rpc_lock`
  - 管 executor transport RPC

这部分是本轮后补修复，不是初始三笔功能 commit 的一部分。

落地 commit:

- `47409a96 Split diffusion engine rpc and state locks`

#### 5.6.2 并发语义与不变量

当前已成立的不变量:

- scheduler 状态更新不再与长时间 executor RPC 共享同一把锁
- `close()` 可以在 stepwise RPC 期间推进状态收敛并唤醒 waiter
- `abort()` 对 waiting / running 请求采用不同收敛路径

#### 5.6.3 请求提交流程的锁边界

当前行为:

- request 入队和 waiter 注册在 `_engine_lock` 下完成
- stepwise RPC 在 `_rpc_lock` 下执行
- terminal output 回写在 `_engine_lock` 下完成

#### 5.6.4 scheduler thread 的锁边界

当前行为:

- scheduler thread 在 `_engine_lock` 下取调度输出
- 在 `_rpc_lock` 下执行 `execute_stepwise()`
- 回到 `_engine_lock` 完成 `update_from_output()` 和 waiter 发布

#### 5.6.5 `abort()`、`collective_rpc()`、`close()` 的锁规则

当前状态:

- `abort()` 已实现 step-level 收敛逻辑
- `close()` 已实现 pending waiter 解阻塞
- `collective_rpc()` 通过分离后的 `_engine_lock` / `_rpc_lock` 继续工作

### 5.7 abort 闭环

已完成:

- waiting 请求: 立即 finished-aborted
- running 请求: 标记 `abort_pending`，在下一次 step 边界完成收敛
- async 入口的旧 `abort()` 覆盖 bug 已修复

落地 commit:

- `366fdb31 Wire step-level diffusion engine loop`
- `1abe14df Fix async diffusion abort override`

### 5.8 executor / worker / pipeline

已完成:

- executor 新增 `execute_stepwise()` all-rank transport helper
- multiprocess executor 可以返回 `RunnerOutput`
- worker 包级导出改成 lazy import，避免 scheduler / worker 初始化导入环
- `layerwise_backend.py` 打开 postponed annotations，避免平台类型在 import 时求值炸掉

落地 commit:

- `8646ea21 Add step-level diffusion scheduler kernel`
- `14427b41 Fix diffusion worker import initialization`

## 6. 策略迁移顺序

当前实际进度:

- `fcfs`: 已落地
- `sjf`: 已落地
- `sjf_aging`: 已落地
- `sjf_aging_guarded`: 已落地
- `sjf_aging_guarded_tail`: 已落地
- `p95-first`: 已落地
- `guarded`: 已由 `sjf_aging_guarded` / `sjf_aging_guarded_tail` 覆盖首批迁移
- `fusion`: 未开始

判断:

- 当前主干已完成第一批策略迁移，说明 `scheduler -> policy hook -> execution state` 这条承载结构已经成立
- 后续优先级转为：
  - `p95-fusion`
  - `p95-first-deadline`
  - `p95-bucket-sjf`
  - 多请求 / soak / fairness 验证

## 7. PR 拆分建议

原方案建议 `PR1 -> PR4` 四段式推进；本轮实际为了降低回滚风险，拆成了 6 个最小 commit:

| 顺序 | Commit | 说明 |
| --- | --- | --- |
| 1 | `21a25d29` | config + scheduler contract + CLI 入口 |
| 2 | `8646ea21` | scheduler kernel + policy + executor stepwise transport |
| 3 | `366fdb31` | engine step loop + abort + batch rejection |
| 4 | `1abe14df` | async diffusion abort 覆盖修复 |
| 5 | `14427b41` | worker import initialization 修复 |
| 6 | `47409a96` | engine rpc/state 锁拆分修复 |

与原 PR 拆分关系:

- 原 `PR1` 基本对应 `21a25d29`
- 原 `PR2` 的 scheduler kernel 部分对应 `8646ea21`
- 原 `PR3` 基本对应 `366fdb31`
- 原 `PR4` 尚未开始
- `1abe14df` / `14427b41` / `47409a96` 属于实现期暴露出来的真实阻塞修复

## 8. 测试与验收

### 8.1 必测

代码级验证:

- `PYTHONNOUSERSITE=1 /home/tianzhu/.conda/envs/vllm-omni-v18/bin/python -m pytest -q tests/entrypoints/test_async_omni_diffusion_config.py tests/diffusion/test_diffusion_scheduler.py tests/diffusion/test_multiproc_engine_concurrency.py`
- 结果: `41 passed, 17 warnings in 17.71s`

真实服务验证环境:

- GPU: `gpu0`
- 模型: `/home/tianzhu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6`
- 基准环境: `/home/tianzhu/.conda/envs/vllm-omni-v18`
- 基准代码路径: `/home/tianzhu/vllm-omni-baseline` `911be0d4`
- 实验代码路径: `/home/tianzhu/vllm-omni` `47409a96`
- benchmark 脚本: `benchmarks/diffusion/diffusion_benchmark_serving.py`
- Dataset C 口径: `qwen_image_serving_performance.md` 中的 `Mix Resolution`
- 实际 benchmark 参数:
  - `--dataset random`
  - `--task t2i`
  - `--num-prompts 10`
  - `--max-concurrency 1`
  - `--warmup-requests 0`
  - `--enable-negative-prompt`
  - `--random-request-config '[{"width":512,"height":512,"num_inference_steps":20,"weight":0.15},{"width":768,"height":768,"num_inference_steps":20,"weight":0.25},{"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},{"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}]'`

基准组 / 实验组对比:

| 指标 | 基准组 `911be0d4` | 实验组 `47409a96` | 差值 |
| --- | --- | --- | --- |
| completed requests | `10` | `10` | `0` |
| failed requests | `0` | `0` | `0` |
| QPS | `0.0530003` | `0.0534005` | `+0.0004002` |
| Mean latency | `18.8677s` | `18.7263s` | `-0.1414s` |
| P50 latency | `14.7613s` | `14.5960s` | `-0.1653s` |
| P95 latency | `49.3530s` | `49.0537s` | `-0.2993s` |
| P99 latency | `67.9712s` | `67.5487s` | `-0.4225s` |
| Peak memory max in benchmark JSON | `68724 MB` | `68724 MB` | `0` |
| Peak memory max from `nvidia-smi` sampling | `69789 MB` | `69789 MB` | `0` |

结论:

- 两组 10/10 全部成功
- 吞吐、延迟、峰值显存均一致
- 当前改动没有引入可见性能回归

工件路径:

- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/baseline_metrics.json`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/baseline_benchmark.log`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/baseline_gpu_mem.csv`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/experiment_metrics.json`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/experiment_benchmark.log`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/experiment_gpu_mem.csv`
- `benchmarks/diffusion/perf_runs/20260401_step_level_mvp/experiment_server_stdout.log`

### 8.2 建议新增测试

下一轮建议补的测试:

- 多请求排队场景下的 step-boundary fairness / FCFS 顺序验证
- HTTP 层 `abort()` 对 waiting / running 请求的端到端测试
- `close()` 与长时间 stepwise RPC 并发收敛的集成测试
- step-level serve 路径的更长时间 soak test

### 8.3 DoD

当前已经满足的 DoD:

- 可编译
- 定向单测通过
- 真实 `vllm serve` 可启动
- step-level 功能 smoke 跑通
- 基准组 / 实验组 10 条请求 benchmark 完成
- 修改后代码性能与基准代码一致

当前未满足的扩展 DoD:

- `RuntimeProfileEstimator`
- 非 `fcfs` 策略
- `batch_size > 1`

## 9. 风险与回滚

### 主要风险

- 新迁入策略目前只完成定向单测，还没有真实 benchmark/soak 级回归
- 仍未恢复独立 `RuntimeProfileEstimator service`
- step-level 路径尚未覆盖 batch / multi-prompt
- 本轮 perf 工件里缺少 baseline server stdout 独立文件；基准组仍保留了 benchmark log、metrics、GPU 显存采样，但若后续需要逐行 server 启动日志，需要复跑补齐

### 回滚路径

若只回滚 MVP 改动，可按逆序回滚以下 commit:

1. `47409a96`
2. `14427b41`
3. `1abe14df`
4. `366fdb31`
5. `8646ea21`
6. `21a25d29`

这 6 个 commit 已按单功能最小粒度拆开，具备独立回滚性。

## 10. 最终拍板

截至 `2026-04-01`，可以认为:

- `v16-base` step-level 最小能力已经以 `v18-base` 分层风格成功落地
- 当前交付物满足“最小可落地版本 + 真实服务验证 + 性能一致性验证”的目标
- 若后续继续推进，优先顺序应为:
  1. `p95-fusion` / deadline / bucket 系列策略迁移
  2. 新迁入策略的真实 benchmark / soak / fairness 验证
  3. 多请求公平性 / soak test
  4. batch / multi-prompt 扩展

## 11. `2026-04-02` Benchmark 迁移补充

### 11.1 背景与决策

- step-level 主链路已完成，但 `benchmarks/diffusion` 仍缺 `v16-base` 中对 warmup、SLO 估时、输出保存、单实例 Wan 压测脚本的增强。
- 当前 `v18-base` 下 `/v1/videos` 已经升级为异步 job API 语义，因此 benchmark 端不能回退到 `v16-base` 的同步请求模型。
- 当前 `v18-base` 下 `vllm_omni/global_scheduler` runtime 还未完成迁移；工作树中只有目录骨架与 `policies/`，缺少 `config.py` / `server.py` / `router.py` 等核心模块，因此本轮不恢复 global-scheduler orchestration 脚本。

### 11.2 本轮已迁移内容

- `benchmarks/diffusion/backends.py`
  - `RequestFuncInput` 新增 `estimated_cost_s`
  - `RequestFuncOutput.response_body` 放宽为 `Any`，兼容视频二进制内容落盘
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
  - 新增 `--warmup-request-config`
  - warmup 支持按 request profile 构造，并支持 `weight` 驱动的确定性展开
  - warmup 结果可同时推导 `slo_ms` 与 `estimated_cost_s`
  - `estimated_cost_s` 默认注入 request payload；`--inject-scheduler-slo` 额外注入 `slo_ms` / `slo_target_ms`
  - 新增 `--save-output-dir`
  - `aiohttp` client session 显式设置长超时，避免长视频 benchmark 被默认超时提前截断
  - 保持当前 task/backend 强校验、Poisson arrival、stage duration 汇总逻辑不变
- `benchmarks/diffusion/README.md`
  - 补充 warmup config、scheduler 字段注入、结果保存说明
- `benchmarks/diffusion/scripts/run_wan_sp4_cfg2_hsdp_rps_bench.sh`
  - 恢复单实例 Wan RPS sweep 脚本
  - 同时支持 `fixed_duration` / `fixed_num_prompts`
  - 修复 `summary.csv` 的列数不匹配问题

### 11.3 本轮明确保留的 `v18-base` 行为

- `/v1/videos` 继续采用异步 job -> poll -> content 的 benchmark 语义
- task/backend 非法组合继续显式报错，不再自动改写 backend
- `iter_requests()` 继续使用 Poisson 到达流；本轮没有切到固定间隔发包
- stage duration 聚合逻辑保持启用

### 11.4 本轮未迁移项

- `benchmarks/diffusion/scripts/global_instance_scheduler_v2/*`
- `benchmarks/diffusion/scripts/run_global_scheduler_benchmark*.sh`
- `benchmarks/diffusion/scripts/README_global_instance_scheduler.md`

原因:

- 当前 global scheduler runtime 尚未在 `v18-base` 中完成迁移，恢复这批脚本会直接依赖缺失模块
- 当前 MVP 仍只保证单实例 / 本地 step-level 主路径，global scheduler 合并应作为下一阶段独立任务推进

### 11.5 验证

已完成:

- `python3 -m py_compile benchmarks/diffusion/backends.py benchmarks/diffusion/diffusion_benchmark_serving.py`
- `bash -n benchmarks/diffusion/scripts/run_wan_sp4_cfg2_hsdp_rps_bench.sh`
- `/home/tianzhu/.conda/envs/vllm-omni-v18/bin/python benchmarks/diffusion/diffusion_benchmark_serving.py --help`

结果:

- Python 语法检查通过
- shell 语法检查通过
- benchmark CLI 新参数注册正确

附注:

- 系统默认 `python3` 环境缺少 `aiohttp`
- benchmark 运行仍应继续使用 `/home/tianzhu/.conda/envs/vllm-omni-v18`

### 11.6 当前结论

- step-level MVP 主链路与本地实例 benchmark 客户端能力已经在同一轮收口
- `estimated_cost_s` 注入路径已打通，为后续非 `fcfs` 策略和 global scheduler 恢复保留了 benchmark 侧接口
- global scheduler runtime 与 orchestration 脚本迁移仍是后续独立里程碑

## 12. `2026-04-02` Global Scheduler 迁移落地

### 12.1 本轮确认后的边界

- 全局策略只保留:
  - `min_queue_length`
  - `round_robin`
  - `short_queue_runtime`
- worker 侧只接受并透传:
  - `slo_ms`
  - `estimated_cost_s`
- 不恢复 `--diffusion-engine-max-concurrency`
- orchestration 只保留并强制 worker 启动参数:
  - `--diffusion-scheduler-backend step_level_request_scheduler`
  - `--diffusion-enable-step-chunk`

补充决定:

- `global scheduler` 按 pure routing 落地，不在 scheduler 侧等待请求
- runtime state 只做 bookkeeping，不维护 scheduler-side waiting queue
- worker 内部是否等待、如何执行，继续由 worker 本地 scheduler 决定

### 12.2 本轮已迁移内容

- `vllm_omni/global_scheduler/*`
  - 恢复 `config.py`、`types.py`、`runtime_profile.py`、`router.py`、`state.py`
  - 恢复 `lifecycle.py`、`process_controller.py`、`server.py`
  - 恢复 `__init__.py`、`README.md`、`README_zh.md`
- `vllm_omni/global_scheduler/policies/*`
  - 恢复 `policy_base.py`
  - 恢复 `algorithm_policy_router.py`
  - 恢复 `runtime_estimator.py`
  - 恢复 `min_queue_length.py`
  - 恢复 `round_robin.py`
  - 恢复 `short_queue_runtime.py`
- worker metadata 透传
  - `vllm_omni/entrypoints/openai/protocol/images.py`
  - `vllm_omni/entrypoints/openai/protocol/videos.py`
  - `vllm_omni/entrypoints/openai/api_server.py`
  - `vllm_omni/entrypoints/openai/serving_video.py`
  - `vllm_omni/entrypoints/openai/serving_chat.py`
- benchmark / orchestration 精简版恢复
  - `benchmarks/diffusion/scripts/global_instance_scheduler_v2/orchestrate.py`
  - `benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh`
  - `benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh`
  - `benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml`
  - `benchmarks/diffusion/scripts/global_instance_scheduler_v2/README.md`
  - `benchmarks/diffusion/scripts/README_global_instance_scheduler.md`
  - `benchmarks/diffusion/scripts/run_global_scheduler_benchmark.sh`
  - `benchmarks/diffusion/scripts/run_global_scheduler_benchmark_one_shell.sh`
- 测试恢复
  - 恢复 `tests/global_scheduler/*`
  - 新增 / 更新 `slo_ms` 与 `estimated_cost_s` 透传测试

### 12.3 与旧方案相比的明确变化

- 不再从 worker launch args 解析 `diffusion_engine_max_concurrency`
- 不再让 `global scheduler` 等待“容量释放”后再转发
- `min_queue_length` 与 `short_queue_runtime` 仅基于当前已路由请求的 bookkeeping 打分
- orchestration 不再依赖:
  - `chunk_preemption`
  - `chunk_budget`
  - `diffusion_engine_max_concurrency`

### 12.4 本轮未纳入项

- `fcfs`
- `estimated_completion_time`
- worker 字段 `slo_target_ms`
- worker 字段 `deadline_ts`
- 实例内复杂策略:
  - `p95-fusion`
  - `p95-first-deadline`
  - `p95-bucket-sjf`
  - `fusion`
- `api_server.py` request-id / arrival / finish 日志增强

### 12.5 验证

已完成:

- `PYTHONNOUSERSITE=1 /home/tianzhu/.conda/envs/vllm-omni-v18/bin/python -m pytest -q tests/global_scheduler tests/entrypoints/openai_api/test_image_server.py tests/entrypoints/openai_api/test_video_server.py tests/entrypoints/openai_api/test_serving_chat_sampling_params.py`
- `PYTHONNOUSERSITE=1 /home/tianzhu/.conda/envs/vllm-omni-v18/bin/python -m py_compile vllm_omni/global_scheduler/*.py vllm_omni/global_scheduler/policies/*.py benchmarks/diffusion/scripts/global_instance_scheduler_v2/orchestrate.py`
- `bash -n benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh benchmarks/diffusion/scripts/run_global_scheduler_benchmark.sh benchmarks/diffusion/scripts/run_global_scheduler_benchmark_one_shell.sh`
- `PYTHONNOUSERSITE=1 /home/tianzhu/.conda/envs/vllm-omni-v18/bin/python -c "from vllm_omni.global_scheduler.config import load_config; from vllm_omni.global_scheduler.server import create_app; cfg=load_config('benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml'); app=create_app(cfg); print(len(app.routes))"`

结果:

- 定向 pytest 通过，`148 passed`
- Python 语法检查通过
- shell 语法检查通过
- `load_config + create_app` smoke 通过，创建出的 app route 数为 `16`

### 12.6 当前结论

- `v18-base` 已具备最小可用的 global scheduler runtime
- 3 个目标全局策略已恢复并可测试
- `estimated_cost_s` 从 benchmark / client -> global scheduler -> worker 的链路已收口
- `slo_ms` 已在 worker 侧接受并透传，为后续 SLO-aware 策略保留接口
- 当前实现刻意保持 pure routing 语义，避免把请求堆在 global scheduler 侧

### 12.7 `2026-04-06` 文档与模板收尾补充

本轮没有新增 runtime 行为改动，主要完成 benchmark/orchestration 与 global scheduler 文档层的对齐收尾。

已完成:

- `benchmarks/diffusion/scripts/global_instance_scheduler_v2/orchestrate.py`
  - 新增 `BENCHMARK_WARMUP_REQUEST_CONFIG -> benchmark.warmup_request_config` 透传
  - 使 orchestrator 能把 `--warmup-request-config` 传给 `diffusion_benchmark_serving.py`
- benchmark 模板补齐
  - `single_instance.qwen.yaml` 改为显式 warmup profile
  - 新增 `single_instance.wan2_2.yaml`
  - 删除模板中无效的 `benchmark.output_file`，统一改为由 orchestrator 接管 metrics 输出路径
- benchmark 文档补齐
  - 重写 `benchmarks/diffusion/scripts/global_instance_scheduler_v2/README.md`
  - 新增 / 重写 `benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md`
  - 新增 `benchmarks/diffusion/README_zh.md`
- global scheduler 模块文档补齐
  - 重写 `vllm_omni/global_scheduler/README.md`
  - 重写 `vllm_omni/global_scheduler/README_zh.md`
  - 明确 pure routing 语义、接口、配置、runtime profile、lifecycle 和已知限制

这批收尾带来的直接结果:

- benchmark/orchestration README 不再把 `fixed_duration` 误写成“硬截止发送时间”，而是明确说明当前只是“按目标时长推导请求数”
- warmup 的构造方式在模板、orchestrator 和 README 之间已对齐
- `Wan2.2` 和 `Qwen` 两条单实例 global scheduler benchmark 入口都已有可编辑模板
- global scheduler 模块自身已有完整的中英文运行说明，可独立用于配置、部署和调试

## 13. 后续工作

### 13.1 高优先级

- 做真实多实例 smoke
  - 至少验证 `2+` worker 下的 `min_queue_length / round_robin / short_queue_runtime`
  - 覆盖 `Qwen` 与 `Wan2.2` 两类模板
  - 补齐真实 `/instances`、lifecycle、routing header 与 metrics 对账
- 做 benchmark 真实回归
  - 用 `global_instance_scheduler_v2` 跑一轮最小 case/suite
  - 产出一份真实结果样例，验证 README 中的目录结构与命令说明
- 补齐 runtime profile / cost 质量
  - 为目标实例类型准备 `runtime_profile.json`
  - 或确保 benchmark/client 在更多链路上稳定注入 `estimated_cost_s`
  - 否则 `short_queue_runtime` 的收益会被明显削弱

### 13.2 中优先级

- 如果需要严格“固定发送时长”，扩展 `diffusion_benchmark_serving.py`
  - 当前 `fixed_duration` 只是根据 `request_rate * duration` 反推 `num_prompts`
  - 还没有 wall-clock 到点硬停止发送的机制
- 如果 warmup 需要独立数据源，新增：
  - `warmup_dataset`
  - `warmup_dataset_path`
  - 当前 warmup 仍然复用主 benchmark 数据集，只在 profile 层覆盖 shape / steps / frames
- 评估是否从 config schema 中彻底删掉 `benchmark.output_file`
  - 模板层已经移除
  - 但 `GlobalSchedulerConfig.BenchmarkConfig` 里仍保留该字段用于兼容

### 13.3 后续功能扩展

- 继续迁移全局 / 实例内复杂策略：
  - `p95-fusion`
  - `p95-first-deadline`
  - `p95-bucket-sjf`
  - `fusion`
- 评估是否引入真正的 SLO-aware routing
  - 当前 `slo_ms` 已透传
  - 但 global scheduler 还没有消费它
- 如果未来需要更强控制，再讨论是否引入 scheduler-side waiting / admission
  - 当前实现刻意保持 pure routing
  - 不建议在没有新的状态所有权设计前直接回退到“全局等待队列”方案

## 14. `2026-04-06` Step-Level 策略迁移补充

### 14.1 本轮新增落地

- `vllm_omni/diffusion/sched/policy.py`
  - 从 `v16-base` 迁入 `sjf`
  - 从 `v16-base` 迁入 `sjf_aging`
  - 从 `v16-base` 迁入 `sjf_aging_guarded`
  - 从 `v16-base` 迁入 `sjf_aging_guarded_tail`
  - 从 `v16-base` 迁入 `p95-first`
- `vllm_omni/diffusion/sched/interface.py`
  - `DiffusionExecutionState` 新增 dispatch runtime / cumulative execute time 字段
- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 增加 policy lifecycle hook：
    - `on_request_arrival`
    - `on_request_scheduled`
    - `on_step_complete`
    - `on_request_finished`
- `vllm_omni/diffusion/data.py`
  - step-level policy allowlist 从仅 `fcfs` 扩展到 5 个已迁入策略
  - 接受 alias `sjf_aging_guard -> sjf_aging_guarded`
- 文档
  - 更新 `vllm_omni/diffusion/sched/README.md`
  - 更新 `vllm_omni/diffusion/sched/README_zh.md`
  - 更新 `benchmarks/diffusion/scripts/global_instance_scheduler_v2/README.md`
  - 更新 `benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md`

### 14.2 当前语义边界

- 这轮迁移的是 waiting-queue selection 语义，不是把旧 `Stage1Scheduler` 整体搬回
- `sjf_aging_guarded_tail` 当前只迁入 tail sink / hard escape / budget 这组 waiting-queue 逻辑
- 旧实现中的 chunk-budget override、idle-only `3x` chunk 扩张等行为仍未迁入
- `p95-first` 当前保留 learned service-rate / slowdown 主路径，但没有把旧版大规模 CLI 调参面一起恢复

### 14.3 验证

已完成:

- `PYTHONNOUSERSITE=1 /home/tianzhu/.conda/envs/vllm-omni-v18/bin/python -m pytest -q /home/tianzhu/vllm-omni/tests/diffusion/test_diffusion_scheduler.py /home/tianzhu/vllm-omni/tests/entrypoints/test_async_omni_diffusion_config.py`

结果:

- `40 passed`
- 已覆盖：
  - builder / alias
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`
  - config allowlist / reject path

## 15. `2026-04-07` / `2026-04-08` 观测性与 Benchmark 可复现性补充

### 15.1 本轮新增落地

- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - 新增逐 step 观测日志：
    - `[StepSchedule]`
    - `[StepComplete]`
  - 调度日志现在会记录：
    - `turn`
    - `policy`
    - `selected request`
    - `new/resumed`
    - `dispatch_epoch`
    - `progress`
    - `waiting_before`
    - waiting 队列头部摘要
  - step 完成日志现在会记录：
    - `progress old -> new`
    - `finished`
    - `status`
    - 单 step 耗时
    - 累计执行时间
    - 剩余估时
- `tests/diffusion/test_diffusion_scheduler.py`
  - 新增逐 step 日志链路单测
  - 覆盖“首次调度 -> unfinished requeue -> 再次调度 -> 最终完成”的日志闭环
- `vllm_omni/entrypoints/async_omni_diffusion.py`
  - executor worker 数改为环境变量可配置：
    - `VLLM_OMNI_DIFFUSION_EXECUTOR_MAX_WORKERS`
  - 默认值不再固定为 `1`
  - 默认改为按 CPU 数推导的保底/封顶值
  - 新增 executor 入口与完成日志：
    - `[AsyncDiffusionExecutorEnqueue]`
    - `[AsyncDiffusionExecutorStart]`
    - `[AsyncDiffusionExecutorDone]`
  - 单请求与 batch 统一收口到带观测的 executor submit path
- `tests/entrypoints/test_async_omni_diffusion.py`
  - 覆盖 executor worker 数环境变量解析
  - 覆盖 executor lifecycle 日志输出
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
  - 新增 `DEFAULT_ARRIVAL_SEED = 42`
  - `iter_requests()` 改为使用独立 `random.Random(arrival_seed)`
  - 不再直接依赖全局 `random.expovariate(request_rate)` 的隐式初始状态
  - 新增 CLI 参数：
    - `--arrival-seed`
  - benchmark summary 里新增：
    - `Arrival seed`
  - 默认行为改为“Poisson 到达 + 固定 arrival seed”
- `tests/benchmarks/test_diffusion_benchmark_serving.py`
  - 验证默认 arrival seed 固定且不受全局 `random` 影响
  - 验证自定义 arrival seed 可以改变 inter-arrival 序列

### 15.2 本轮问题定位结论

- 针对 `2026-04-07` 旧 benchmark 结果目录的复盘确认：
  - 同 policy / 同 RPS 的大幅波动主要不是请求 mix 漂移
  - `random_request_seed` 已固定请求 profile 序列
  - 真正不固定的是 arrival trace
- 旧 benchmark 的 `iter_requests()` 在修复前使用全局 `random.expovariate(request_rate)` 生成 Poisson 到达间隔
- 由于 benchmark 每次 rerun 都会起新的 Python 进程，而这部分没有显式固定全局 `random` seed，因此同 RPS rerun 之间会得到不同 arrival trace
- 旧 `max_workers=1` 阶段，请求主要在 engine 前排队，因此 end-to-end latency 对 arrival burstiness 非常敏感
- 这解释了为什么旧结果中不仅“不同 policy 之间差异很大”，甚至“同 policy / 同 RPS 在不同 rerun 之间也会明显漂移”

### 15.3 新日志与行为确认

- 在放开 executor worker 数并补齐日志之后：
  - `waiting_before` 不再长期卡在 `1`
  - executor `active` 已出现 `>1`
  - 说明请求不再只是“前一个完全跑完之后，下一个才真正进入 engine”
- 新 `worker0.log` 已观测到真实 step-level 抢占
- 结合新日志可确认：
  - 请求已经按 step-level 进入本地 scheduler
  - `sjf` 不再只是代码语义上的 waiting-queue 排序，而是能在日志中看到真实的 shorter-job 插队片段

### 15.4 验证

已完成:

- `/home/wtz2333/.conda/envs/vllm-18/bin/python -m pytest /home/wtz2333/vllm-omni/tests/diffusion/test_diffusion_scheduler.py -q`
- `/home/wtz2333/.conda/envs/vllm-18/bin/python -m pytest /home/wtz2333/vllm-omni/tests/entrypoints/test_async_omni_diffusion.py -q`
- `/home/wtz2333/.conda/envs/vllm-18/bin/python -m pytest /home/wtz2333/vllm-omni/tests/benchmarks/test_diffusion_benchmark_serving.py -q`

结果:

- `tests/diffusion/test_diffusion_scheduler.py`: `25 passed`
- `tests/entrypoints/test_async_omni_diffusion.py`: `2 passed`
- `tests/benchmarks/test_diffusion_benchmark_serving.py`: `2 passed`
- 当前环境仍有既有 warning：
  - `PytestConfigWarning: Unknown config option: asyncio_mode`
  - 若干 `torch.jit.script_method` / swig 相关 `DeprecationWarning`

### 15.5 当前结论

- step-level / instance policy 的可观测性已经足以支持后续真实时序分析
- executor 入口不再被默认单 worker 人工串死，单实例 step-level 行为现在可以在 worker 日志中直接观察
- benchmark arrival trace 默认已可复现，后续做同 RPS 多 policy 对比时，不再受“到达 seed 未固定”这一类噪声干扰
- 若后续要继续提升实验公平性，剩余增强项已经从“固定 arrival seed”收敛为：
  - 是否持久化完整 arrival trace
  - 是否持久化 `estimated_cost_s`
  - 是否提供 trace replay 模式
