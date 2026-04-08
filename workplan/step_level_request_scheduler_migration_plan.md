# `v16-base` Step-Chunk / 实例内调度 到 `v18-base` 的重构式迁移方案（新版）

## Summary

- 目标不是把历史 `Stage1Scheduler` 直接搬进 `v18-base`，而是按 `v18-base` 当前的 `SchedulerInterface -> RequestScheduler -> engine/executor` 分层，重构出一个新的 `StepLevelRequestScheduler`，同时保留原 `Stage1Scheduler` 的策略能力。
- 首次上游合入按 **experimental / opt-in** 设计，不作为稳定公开接口承诺。默认仍是 `request_scheduler + step_chunk=False`。
- 新方案直接解决当前阻塞项：
  - 不再让 engine 假设“只会执行 `scheduled_new_reqs[0]`”
  - 先定义 scheduler-owned execution state，再接 chunk lifecycle
  - 只支持清晰、可测试的配置组合，避免半定义模式进入上游
- 迁移目标是 **策略等价、结构重构**：保住原 `Stage1Scheduler` 的 `fcfs / sjf / p95-first / p95-fusion / guarded / defer-budget` 等策略语义，但落到 `v18-base` 原生抽象里。

## Key Changes

### 1. 先把公开接口和支持矩阵收紧成“上游可接受”的版本

- 公开配置只引入：
  - `diffusion_scheduler_backend`
  - `instance_scheduler_policy`
  - `diffusion_enable_step_chunk`
  - `diffusion_enable_chunk_preemption`
  - `diffusion_chunk_budget_steps`
  - `instance_runtime_profile_path`
  - `instance_runtime_profile_name`
- `diffusion_scheduler_backend` 仅支持：
  - `request_scheduler`（默认）
  - `step_level_request_scheduler`（experimental）
- 上游首版只正式支持两组组合：
  - `request_scheduler + step_chunk=False`
  - `step_level_request_scheduler + step_chunk=True`
- 下面两组在首版直接做 config error，不做 warning：
  - `request_scheduler + step_chunk=True`
  - `step_level_request_scheduler + step_chunk=False`
- 当 backend 是 `request_scheduler` 时，`instance_scheduler_policy != fcfs` 直接报配置错误；不允许“传了 policy 但 silently ignore”。

### 2. 先补 contract，再补算法

- 在 scheduler 层新增一个正式的 scheduler-owned execution state 类型，建议命名为 `DiffusionExecutionState`，按 `sched_req_id` 持有：
  - `executed_steps`
  - `total_steps`
  - `planned_chunk_budget_steps`
  - `estimated_runtime_s`
  - `force_run_to_completion`
  - `last_dispatch_epoch`
  - `terminal_reason`
- `OmniDiffusionRequest` 只保留必要镜像字段；真正的执行进度真相源迁到 scheduler-owned state。
- `DiffusionRequestContext` 继续归 executor/model-runner 持有，不把 context handle 塞进 scheduler state。
- 在 `SchedulerInterface` 现有 contract 基础上，不新增第二套 scheduler 抽象；新 backend 必须直接实现当前 contract。

### 3. 用现有 stepwise worker 路径，不再另造一套执行 contract

- 复用 repo 里已经存在的 stepwise worker/model-runner 能力，新增 executor 级 API：
  - `add_req(request)` 继续服务旧路径
  - `execute_stepwise(scheduler_output)` 作为新路径唯一执行入口
- engine 改为按 `DiffusionSchedulerOutput` 分支：
  - `request_scheduler` 路径继续走当前 request-mode
  - `step_level_request_scheduler` 路径统一走 `schedule -> executor.execute_stepwise -> scheduler.update_from_output`
- `update_from_output()` 的新 contract 必须显式支持两类结果：
  - terminal completion / error / abort
  - unfinished continuation（更新 `executed_steps`、重新入 waiting、保留 context）
- engine 不再直接假设 `scheduled_new_reqs[0]`；它必须能消费 `scheduled_cached_reqs`。

### 4. 把原 `Stage1Scheduler` 拆成三层后再迁策略

- `RequestScheduler`
  - 保留为宿主生命周期状态机
  - 负责 id、waiting/running/finished、finish/preempt/cleanup 基础行为
- `RequestSelectionPolicy`
  - 新增策略抽象，只负责 waiting 选择
  - `fcfs` 也迁成 policy，保证默认路径和扩展路径结构一致
- `StepLevelRequestScheduler`
  - 复用 `RequestScheduler`/`_BaseScheduler` 的主状态机
  - 组合 `RequestSelectionPolicy`、`RuntimeProfileEstimator`、chunk lifecycle hooks
- `ChunkLifecycleController`
  - 不单独暴露为公开 backend
  - 作为 scheduler 内部组件，负责 unfinished requeue、chunk budget 规划、preemption-to-resume 语义
- 不保留 `Stage1SchedulerAdapter` 作为最终目标；若过渡期需要兼容层，只允许作为临时内部实现，不进入公开命名。

## Migration Phases

### Phase 1 — 最小 scaffold，只立公共宿主

- 只在 `request/context/config/AsyncOmniEngine diffusion stage builder` 落最小必要字段。
- 不提前加入任何算法专属参数。
- 这一阶段的落点以 `AsyncOmniEngine._create_default_diffusion_stage_cfg()` 为主，不再把 `async_omni.py` 视为 diffusion config 主入口。
- 验收：
  - 默认行为不变
  - config 只表达 backend/step-chunk/runtime-profile 基础能力
  - 无算法专属参数泄漏到公开 config 面

### Phase 2 — 先落“可被上游接受”的 contract kernel

- 引入 `RuntimeProfileEstimator`
- 引入 `DiffusionExecutionState`
- 给 executor 增加 `execute_stepwise()`，内部复用现有 worker stepwise 路径
- 补 `request_scheduler` 现有 contract 测试，新增 `cached request` / `unfinished` contract 测试
- 这一阶段仍不接入策略调度，只把 chunk continuation contract 打通

### Phase 3 — 引入 `StepLevelRequestScheduler`，先做 `fcfs` 等价闭环

- 新 backend 首先只实现 `fcfs` policy，要求行为与旧 `Stage1Scheduler(fcfs)` 在可观察语义上等价：
  - queue order
  - unfinished requeue
  - metrics / request lifecycle timestamps
  - abort / error normalization
- 这一阶段只接入最小 runtime estimate 读取，不引入所有 policy 参数
- 验收标准是：`step_level_request_scheduler + step_chunk=True` 路径可稳定跑通，且默认路径无回归

### Phase 4 — 按策略族逐批迁移，不一次性全上

- 策略迁移顺序固定为：
  1. `fcfs`
  2. `sjf` / `sjf_aging` / guarded 系列
  3. `p95-first` / `p95-first-deadline`
  4. `p95-bucket-sjf`
  5. `type_fifo_defer_budget`
  6. `slack_hybrid`
  7. `p95-fusion`
- 每迁入一类策略，才新增对应参数、校验和测试；不允许“参数先埋、算法以后再说”。
- 旧 `test_stage1_scheduler.py` 的能力验证要拆成：
  - scheduler contract tests
  - policy behavior tests
  - chunk lifecycle tests
- 策略验收以“语义等价”而不是“代码复用比例”为标准。

### Phase 5 — engine/serving 收尾并准备上游 PR

- engine 仅负责选择 backend 和调度循环，不理解策略内部细节。
- serving 层最后再透传 scheduler-facing metadata。
- PR 对外表述固定为：
  - experimental opt-in backend
  - default path unchanged
  - no behavior change for existing users
- 所有历史 `Stage1Scheduler` 命名从公开面移除，只在迁移说明/内部注释中保留“历史来源”含义。

## Test Plan

- Contract tests
  - `SchedulerInterface` 对 `new/cached/unfinished/terminal/abort/preempt` 的统一行为
  - `request_id <-> sched_req_id` 映射与 cleanup
  - `DiffusionExecutionState` 与 context 生命周期一致性
- Engine/executor tests
  - `request_scheduler + step_chunk=False` 默认路径回归
  - `step_level_request_scheduler + step_chunk=True` 完整闭环
  - cached continuation、late output after abort、repeated finish、context cleanup
- Policy parity tests
  - 从历史 `test_stage1_scheduler.py` 迁出同名语义用例
  - 每个策略族单独建 tests，只验证该策略真正承诺的行为
- Observability tests
  - backend/policy/chunk_count/preempt_count/queue_wait/executed_steps/cleanup_failures 日志或 metrics 存在
- Rejection tests
  - 不支持的 config 组合直接失败
  - backend 与 policy 不匹配时直接失败

## Assumptions and Defaults

- `step_level_request_scheduler` 首版定位为 experimental，不承诺稳定公开兼容。
- 迁移优先级是“`v18-base` 原生结构正确”高于“最大化复用旧 `Stage1Scheduler` 代码”。
- `RequestScheduler` 继续作为宿主状态机；新 backend 是其体系内扩展，不是平行的旧式调度器回灌。
- 只有在 `fcfs` 闭环稳定后，才允许继续迁移更复杂策略。
- 若某个历史策略无法在 `v18-base` contract 下无歧义表达，优先补 contract / state model，不接受临时绕过式实现进入上游。
