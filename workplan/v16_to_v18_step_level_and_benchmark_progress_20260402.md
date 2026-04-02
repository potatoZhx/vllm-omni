# `v16-base` Step-Level 主闭环 + Benchmark 迁移合入 `v18-base` 的当前工作进度

> 对齐文档: `workplan/v16_to_v18_step_level_minimal_landing_plan.md`
>
> 进度截点: `2026-04-02`
>
> 当前代码基线: `v18-base` `5fa15e6d` + benchmark migration working tree

## 0. 结论先行

当前最小可落地版本已经在 `v18-base` 上完成主闭环，并完成了代码级回归、真实 `vllm serve` 功能 smoke、以及基准组 / 实验组的 10 条请求性能一致性验证。

截至 `2026-04-02`，本轮又补齐了 `benchmarks/diffusion` 下与 step-level 实验直接相关的 benchmark client、README 和单实例 Wan RPS 脚本迁移，使本地实例级实验入口与 v16-base 的常用 benchmark 能力重新对齐。

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
- 只保证单请求、非 batch 的 step-level 主路径
- 已验证修改后代码与 `v18-base` 原生基准代码性能一致

本轮未进入实现范围的项:

- `RuntimeProfileEstimator`
- `sjf / p95-first / guarded / fusion` 等复杂策略
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
- `sjf / p95-first / guarded / fusion` 等策略批量迁移
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
  - `RequestSelectionPolicy` 抽象和 `FCFSSelectionPolicy`
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

- `instance_scheduler_policy` 首版只接受 `fcfs`
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
- `build_request_selection_policy()` 首版只开放 `fcfs`

当前状态:

- policy 框架已落地
- 复杂策略仍未开始迁移

### 5.5 runtime estimator

当前状态:

- 未实现
- 本轮明确 defer

原因:

- MVP 目标优先保证 step-level contract、engine loop、abort、真实服务闭环
- `fcfs` 首版不依赖 runtime estimator 也可形成稳定最小闭环

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
- `sjf`: 未开始
- `p95-first`: 未开始
- `guarded`: 未开始
- `fusion`: 未开始

判断:

- 只有在 `RuntimeProfileEstimator` 落地后，后续策略迁移才具备明确收益
- 当前主干已具备继续迁移策略的最小承载结构

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

- 当前只有 `fcfs`，策略扩展能力尚未进入生产验证
- `RuntimeProfileEstimator` 未实现，后续 `sjf / p95-first` 等策略仍缺依赖
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
  1. `RuntimeProfileEstimator`
  2. `sjf / p95-first` 等策略迁移
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
  - `sjf`
  - `p95-first`
  - `guarded`
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
