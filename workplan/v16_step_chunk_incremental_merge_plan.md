# `v16-base` Step-Chunk / 实例内调度 渐进式 Merge 方案

## 目标

- 只聚焦 `v16-base` 的第一部分改动：
  - `step chunk`
  - 实例内调度
- 暂时把 `global scheduler` 视为独立、低冲突、可后并的模块
- merge 策略不是“一次性把 `v16-base` 整体 merge 进来”，而是把实例内调度拆成多轮可验证的增量集成

## 核心原则

- 以 `v18-base` 的结构为主干，尽量把 `v16-base` 的能力“移植”进去，而不是机械保留旧结构
- 对冲突最大的 pipeline / engine 文件，优先做“语义合并”，不要追求文本级最小 diff
- 每一轮 merge 都必须满足三个条件：
  - 能 import / 编译
  - 能通过一组针对性测试
  - 默认行为仍尽量接近 `v18-base`
- `global scheduler` 单独处理，不要在前几轮里掺进去增加噪音

## 兼容性目标

- `step chunk` 必须是“新增能力”，不是“替换原能力”
- 当 `step chunk` 关闭时：
  - 默认行为应尽量保持 `v18-base` 现有语义
  - 原有 run-to-completion 路径必须仍可工作
  - 现有 OpenAI / diffusion 基本调用方式不应被破坏
- 当 `step chunk` 开启时：
  - 才进入分步执行、chunk budget、preemption、实例内调度逻辑
- 能通过配置开关安全回退：
  - `diffusion_enable_step_chunk=False`
  - `diffusion_enable_chunk_preemption=False`
- 后续接入来自 `v16-base` `Stage1Scheduler` 设计、但按 `v18-base` 逻辑重构的新 scheduler backend 时：
  - 原有 scheduler 执行路径也必须保留
  - 不应一上来就让所有请求强制走新 backend
  - 应允许通过配置选择“旧 scheduler 路径”或“新 policy-driven scheduler 路径”
- merge 过程中如果某一轮无法同时满足“新功能生效”和“旧行为保留”，优先保住旧行为，再继续分阶段补新功能

## 具体兼容策略

- 以 `feature flag` 保护新行为，而不是直接改默认控制流
- `forward()` / `generate()` 等主入口优先保留原 run-to-completion 逻辑
- scheduler 集成也走 feature flag / policy gate，而不是直接覆盖默认 scheduler
- 新增 `prepare_generation / step_generation / finalize_generation` 时：
  - 把它们作为扩展路径加入
  - 不直接删除旧路径，直到新路径完全验证稳定
- request / config 增字段时：
  - 新字段应有安全默认值
  - 不应要求老请求方必须传新参数
- serving 层新增调度元数据时：
  - 只在用户显式传入时写入
  - 不改变普通请求的既有解析逻辑
- executor / engine 接入新 policy-driven scheduler backend 时：
  - 先保留 `v18-base` 原 scheduler / queue / dispatch 路径
  - 再新增一条新 backend 控制分支
  - 明确默认走哪条路径，并保留回退开关

## Scheduler 选择开关设计

### 设计目标

- 明确区分两类概念：
  - `scheduler backend`：到底走哪条 scheduler 执行路径
  - `scheduler policy`：在选定 scheduler backend 后，内部采用什么排队 / 选择策略
- 不能用 `instance_scheduler_policy` 直接兼任“路径选择开关”
  - 因为它本质上是新 policy-driven request scheduler 内部的 policy 选择
  - 不是 `RequestScheduler` 和新 scheduler backend 之间的总开关

### 命名决策

为了让后续落地更符合 `v18-base` 的整体设计逻辑，建议把历史实现名与目标落地名明确拆开：

- `Stage1Scheduler`
  - 只作为 `v16-base` 历史实现 / 来源概念名保留
  - 不作为 `v18-base` 最终落地类名
- `StepLevelRequestScheduler`
  - 作为在 `v18-base` 中真正落地的新类名
  - 表达它仍然是 `RequestScheduler` 这一抽象族的一员，只是内部支持 policy-driven 选择
- `step_level_request_scheduler`
  - 作为新的 backend 配置值
  - 用来替代讨论期的 `stage1_scheduler` 占位命名

命名理由：

- 避免把“阶段性实验实现名”带进长期主干
- 明确它仍是 request-lifecycle scheduler，而不是一个特殊 stage 组件
- 为后续把 policy layer / chunk lifecycle controller 解耦留下空间
- 更符合 `v18-base` 当前 `RequestScheduler` / `SchedulerInterface` 的命名体系

### 推荐新增字段

在 `OmniDiffusionConfig` 中新增一个独立字段，例如：

- `diffusion_scheduler_backend: str = "request_scheduler"`

推荐可选值：

- `request_scheduler`
  - 表示沿用 `v18-base` 当前默认路径
  - 即 `DiffusionEngine(... scheduler or RequestScheduler())`
- `step_level_request_scheduler`
  - 表示启用按 `v18-base` 逻辑重构后的 policy-driven request scheduler 路径

可选地，也可以接受一个更短的别名层：

- `legacy`
- `policy`

但内部最好统一标准值，避免配置扩散

说明：

- 当前 Phase 1 的代码脚手架中仍临时使用了 `stage1_scheduler` 这个占位值
- 在真正进入 scheduler backend 落地前，应统一重命名为 `step_level_request_scheduler`
- 文档后续都以最终命名为准

### 为什么开关要放在 `OmniDiffusionConfig`

推荐主入口放在 config，而不是散落在 executor / pipeline / serving 层，原因是：

- `DiffusionEngine` 当前就是按 `SchedulerInterface` 注入 scheduler，最适合在 engine 初始化时做 backend 选择
- `async_omni.py` / `cli/serve.py` 只是配置构造入口，适合透传，不适合承载最终选择逻辑
- executor / pipeline 更适合感知“当前 scheduler backend 是什么”，不适合成为唯一的选择源

所以推荐分层：

1. `async_omni.py` / `cli/serve.py`
   - 负责接收和透传 `diffusion_scheduler_backend`
2. `OmniDiffusionConfig`
   - 负责存储、校验默认值
3. `DiffusionEngine`
   - 负责根据该值实例化真正的 scheduler backend
4. `executor`
   - 只根据 engine 已选中的 backend 做兼容处理，不再二次决定

### 状态所有权与一致性矩阵

这是接入 `step chunk` 和新 policy-driven scheduler backend 时必须先定义清楚的部分。

原因：

- 一旦同时引入：
  - request 生命周期字段
  - scheduler-owned state
  - executor / model runner 内部 context
- 如果不先规定“谁是单一状态源”，后续就很容易出现：
  - 双写
  - 漏清理
  - 重复完成
  - 错误回收

建议原则：

- `request` 是外部请求载体，不应成为所有内部调度状态的最终真相
- `scheduler state` 是调度生命周期真相源
- `execution context` 是执行现场真相源
- `request` 上的部分字段可以作为镜像 / 观测字段存在，但不应反向驱动 scheduler state

建议的所有权定义如下：

1. `request_id`
   - 作用：
     - API 层 / 日志 / 用户可见请求标识
   - owner：
     - `OmniDiffusionRequest`
   - 规则：
     - 允许多个 `request_ids`
     - 不作为 scheduler 内部唯一键

2. `sched_req_id`
   - 作用：
     - scheduler 内部唯一调度键
   - owner：
     - `SchedulerInterface` 实现
   - 规则：
     - scheduler 内部队列、状态表、finish/preempt/abort 都以它为主键
     - `request_id -> sched_req_id` 只做映射，不反客为主

3. `waiting / running / preempted / finished`
   - 作用：
     - 请求调度生命周期状态
   - owner：
     - scheduler-owned state
   - 规则：
     - 只允许 scheduler 修改
     - `request.request_state` 如果存在，只作为镜像字段
     - 不允许 executor / runner 直接把 request 标成 finished 来绕过 scheduler

4. `executed_steps`
   - 作用：
     - 表示已完成的 step 数
   - 推荐 owner：
     - scheduler-owned state
   - 备选 owner：
     - execution context
   - 强约束：
     - 必须二选一，不允许 scheduler 和 context 各自独立累计
   - 推荐实现：
     - context 只报告“本次 chunk 实际执行了多少步”
     - scheduler 在 `update_from_output()` 中统一提交到最终状态

5. `max_steps_this_turn` / `chunk budget`
   - 作用：
     - 当前调度轮次允许执行的 step 上限
   - owner：
     - scheduler-owned state
   - 规则：
     - scheduler 负责计算
     - executor / runner 只消费，不自行重写调度预算

6. `finished`
   - 作用：
     - 请求是否已经终态结束
   - owner：
     - scheduler-owned state
   - 规则：
     - worker output / context 只能提供事件
     - 最终是否 finished 由 scheduler 决定并落表

7. `error`
   - 作用：
     - 请求终态失败原因
   - owner：
     - scheduler-owned state
   - 规则：
     - worker / runner / pipeline 只上报错误
     - scheduler 统一决定状态是否进入 `FINISHED_ERROR`

8. `DiffusionRequestContext`
   - 作用：
     - 保存可恢复执行现场
   - owner：
     - executor / model runner
   - 规则：
     - context 的创建、保留、销毁要受 scheduler 事件驱动
     - preempt 时保留
     - finish / abort / unrecoverable error 时销毁

9. `abort`
   - 作用：
     - 终止请求，不再继续调度
   - owner：
     - scheduler
   - 规则：
     - scheduler 标记终态
     - executor / runner 负责 best-effort 清理 context
     - 晚到的 worker output 不得把已 abort 请求重新标成 completed

10. `preempt`
    - 作用：
      - 暂停当前执行并回到可调度状态
    - owner：
      - scheduler
    - 规则：
      - preempt 不等于 abort
      - preempt 后 context 应保留
      - scheduler 状态切回 waiting / preempted

### 推荐同步方向

为了避免多处反向写状态，推荐只允许以下方向：

1. request ingress
   - 生成 `OmniDiffusionRequest`
2. scheduler
   - 为 request 分配 `sched_req_id`
   - 维护 waiting/running/finished
   - 计算 chunk budget
3. executor / runner
   - 根据 scheduler 给出的 budget 执行
   - 返回 chunk 执行结果 / 错误 / 实际执行步数
4. scheduler
   - 在 `update_from_output()` 中统一提交最终状态变更
5. request mirror fields
   - 仅在需要观测时由 scheduler 同步更新

禁止的方向：

- executor 直接决定请求 finished
- context 直接决定请求终态
- request 上镜像字段反向覆盖 scheduler-owned state

### 一致性检查要求

在真正编码前，至少要补一张正式表格，覆盖：

- 字段名
- owner
- 写入者
- 读取者
- 状态迁移事件
- 清理时机
- 是否允许镜像字段存在

如果不先补这张表，Phase 4 很容易在实现过程中把：

- `request_id / sched_req_id`
- `executed_steps`
- `finished / error`
- `preempt / abort`

这几组语义混写，最后导致：

- 重复 finish
- context 残留
- 队列中已完成请求未移除
- abort 后被晚到结果“复活”

### 推荐的控制流

推荐在 `DiffusionEngine.__init__` 附近形成类似如下逻辑：

```python
if scheduler is not None:
    self.scheduler = scheduler
elif od_config.diffusion_scheduler_backend == "step_level_request_scheduler":
    self.scheduler = StepLevelRequestScheduler()
else:
    self.scheduler = RequestScheduler()
```

这个顺序有两个好处：

- 外部显式传入 `scheduler` 时，仍保留测试 / 注入自由度
- 没显式注入时，默认仍走 `request_scheduler`

### 与 `instance_scheduler_policy` 的关系

建议把两者职责完全拆开：

- `diffusion_scheduler_backend`
  - 决定走哪条 scheduler 执行路径
- `instance_scheduler_policy`
  - 只有当 `diffusion_scheduler_backend == "step_level_request_scheduler"` 时才生效

建议规则：

- 当 backend 是 `request_scheduler` 时：
  - 忽略 `instance_scheduler_policy`
  - 最多记录 debug log，不报错
- 当 backend 是 `step_level_request_scheduler` 时：
  - 再读取 `instance_scheduler_policy`
  - 默认值可以继续是 `fcfs`

这样可以避免用户把 `instance_scheduler_policy=p95-first` 误以为会自动切到新的 scheduler backend

### 与 `step chunk` 开关的关系

建议保持两个开关相互独立，不做强绑定：

- `diffusion_enable_step_chunk`
  - 决定请求是否允许分步执行
- `diffusion_scheduler_backend`
  - 决定使用哪条 scheduler 执行路径

推荐兼容策略：

- `request_scheduler + step_chunk=False`
  - 默认旧行为，最安全
- `request_scheduler + step_chunk=True`
  - 允许先局部验证 step API，但仍不启用新的 step-level scheduler
- `step_level_request_scheduler + step_chunk=False`
  - 建议允许配置存在，但初始化时给 warning，提示该组合通常意义不大
- `step_level_request_scheduler + step_chunk=True`
  - 完整启用你要的实例内调度路径

是否要强制校验：

- 初期集成阶段建议只给 warning，不立刻抛错
- 等路径稳定后，再决定是否收紧成强校验

### CLI / Stage Config 设计

推荐在 `cli/serve.py` 和 `async_omni.py` 的 diffusion stage builder 中加入透传：

- CLI 参数建议名：
  - `--diffusion-scheduler-backend`
- 推荐取值：
  - `request_scheduler`
  - `step_level_request_scheduler`

写入 stage config 时放在：

- `stage_cfg["engine_args"]["diffusion_scheduler_backend"]`

不要复用：

- `--instance-scheduler-policy`

因为这个参数的语义应该保留为“step-level request scheduler 的内部 policy”

### 回退设计

为了满足“保留原有功能”，这个开关的默认值必须是：

```python
diffusion_scheduler_backend = "request_scheduler"
```

这意味着：

- 只要用户不显式开启，就继续走 `v18-base` 原路径
- 即使后续 `StepLevelRequestScheduler` 接入完成，也不会自动影响老用户

### 测试建议

接入这个开关后，至少补 4 类测试：

1. 默认配置下：
   - backend 默认为 `request_scheduler`
2. 显式指定 `step_level_request_scheduler` 时：
   - config builder 正确透传
3. `DiffusionEngine` 初始化时：
   - 能根据 backend 选择对应 scheduler 类
4. `instance_scheduler_policy` 在旧 backend 下：
   - 不应偷偷改变 scheduler backend

## 建议分支策略

- 从 `v18-base` 拉一个新集成分支，例如：`v18-stepchunk-integration`
- 每完成一个阶段就单独提交一次
- 每一轮提交信息直接对应一个阶段，便于回退和 bisect

建议提交顺序：

1. `phase1: request/context/config scaffold for step chunk`
2. `phase2: add runtime estimator and stage1 scheduler primitives`
3. `phase2.5: adapt stage1 scheduler to v18 scheduler contract`
4. `phase3: port pipeline/model-runner step APIs`
5. `phase4: wire engine and executor to scheduler loop`
6. `phase5: plumb serving metadata and benchmark compatibility`
7. `phase6: merge global scheduler module`

## 阶段拆分

### Phase 1: 数据结构与配置入口先落地

目标：

- 先让 `v18-base` 拥有 step-chunk 所需的 request state、context、config plumbing
- 这一步先不要求真正调度运行起来

建议文件范围：

- `vllm_omni/diffusion/context.py`
- `vllm_omni/diffusion/request.py`
- `vllm_omni/diffusion/data.py`
- `vllm_omni/entrypoints/async_omni.py`
- `vllm_omni/entrypoints/cli/serve.py`
- `tests/diffusion/test_diffusion_request.py`
- `tests/entrypoints/test_async_omni_diffusion_config.py`
- 如果需要，迁移 `tests/entrypoints/test_omni_stage_diffusion_config.py` 的语义到新入口，而不是直接恢复旧文件

本阶段应完成的事情：

- 给 `OmniDiffusionRequest` 增加实例内调度需要的生命周期字段
- 保留 `v18-base` 的 auto-seed 逻辑
- 引入 `DiffusionRequestContext`
- 把 scheduler / chunk / runtime profile 相关参数先以“最小必要集合”接入 config builder
- 让默认值尽量保持“关闭 step chunk 时不改变行为”

本阶段字段收敛原则：

- **Phase 1 不要一次性引入所有调度算法的初始字段**
- 只引入后续阶段会立刻消费、且属于公共基础设施的字段
- 各类算法专属参数，等对应算法逐个落地并进入可测试状态后，再分阶段加入

Phase 1 推荐只保留的必要字段类型：

- request / context 基础生命周期字段
  - 例如 `arrival_time`、`executed_steps`、`max_steps_this_turn`
- backend 选择字段
  - 例如 `diffusion_scheduler_backend`
- policy 选择字段
  - 例如 `instance_scheduler_policy`
- step-chunk 基础开关与通用 budget 字段
  - 例如 `diffusion_enable_step_chunk`
  - `diffusion_enable_chunk_preemption`
  - `diffusion_chunk_budget_steps`
- runtime profile 基础入口字段
  - 例如 `instance_runtime_profile_path`
  - `instance_runtime_profile_name`

Phase 1 明确不应提前加入的字段类型：

- 某个具体 policy 专属的调参字段
  - 例如 `p95-first` 的 base/min/max/backlog alpha
  - `slack_hybrid` 的 panic threshold
  - `p95-fusion` 的 tail budget / growth / borrowed cap
  - `type_fifo_defer_budget` 的 hard escape multiplier
- 只有在某个算法真正进入实现和测试阶段后才会被消费的细粒度参数

建议执行规则：

- Phase 1 先让 config 能表达“是否启用新 scheduler 路径”和“是否启用 step chunk”
- Phase 2 / Phase 2.5 开始，随着 `RuntimeProfileEstimator`、`RequestSelectionPolicy`、`StepLevelRequestScheduler` 的逐步落地，再把算法专属参数按策略分批接入
- 每引入一组新算法参数，都应同时补对应的单测和默认值约束，而不是只把字段预埋进 config

建议验证：

```bash
pytest -q tests/diffusion/test_diffusion_request.py
pytest -q tests/entrypoints/test_async_omni_diffusion_config.py
```

完成标准：

- 配置入口可接受 step-chunk / scheduler 参数
- 配置字段仍保持最小必要集合，不提前堆积所有算法专属参数
- request/context 类型稳定
- 尚未真正改动 engine 主执行路径
- 关闭 `step chunk` 时不引入行为变化

### Phase 2: 先把 scheduler primitive 独立并稳住

目标：

- 让 runtime profile estimator 和 stage1 scheduler 先以“可单测的独立模块”存在
- 这一步仍尽量少碰 engine 主循环

建议文件范围：

- `vllm_omni/diffusion/runtime_profile.py`
- `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
- `vllm_omni/diffusion/sched/request_selection_policy.py`
- `vllm_omni/diffusion/offline_ideal_scheduler.py`
- `tests/diffusion/test_runtime_profile_estimator.py`
- `tests/diffusion/test_step_level_request_scheduler.py`
- `tests/diffusion/test_request_selection_policy.py`
- `tests/diffusion/test_offline_ideal_scheduler.py`

本阶段应完成的事情：

- 引入 `RuntimeProfileEstimator`
- 引入 `StepLevelRequestScheduler`
- 先验证 policy / queue / cost estimation 的模块级行为
- 不急着把 scheduler 直接接进最复杂的 runtime 路径

建议验证：

```bash
pytest -q tests/diffusion/test_runtime_profile_estimator.py
pytest -q tests/diffusion/test_step_level_request_scheduler.py
pytest -q tests/diffusion/test_request_selection_policy.py
pytest -q tests/diffusion/test_offline_ideal_scheduler.py
```

完成标准：

- scheduler primitive 已经在 `v18-base` 上存在并可测试
- 但生产执行仍可暂时走原始 `run-to-completion`
- 默认路径仍不依赖 scheduler 才能工作
- 旧 scheduler 路径尚未被替换

### Phase 2.5: 先做 RequestScheduler Contract Alignment

目标：

- 在真正接入新 step-level request scheduler 之前，先解决历史 `Stage1Scheduler` 能力与 `v18-base` scheduler 抽象之间的契约差异
- 避免把“后端替换 + 协议迁移 + engine 控制流改写”同时堆到 Phase 4
- 明确最终落地物不是原样保留 `Stage1Scheduler`，而是按 `v18-base` 逻辑收敛后的新类 `StepLevelRequestScheduler`

背景判断：

- `v16-base` 的 `Scheduler` / `Stage1Scheduler` 更偏执行协调器 / 通信调度器
- `v18-base` 的 `SchedulerInterface` / `RequestScheduler` 更偏请求生命周期状态机
- 这说明 `v18-base` 不是只改了 scheduler 名字，而是重新定义了 scheduler 在系统中的职责边界
- 因此历史 `Stage1Scheduler` 不能被视为一个可以直接塞进 `DiffusionEngine` 的“同层 backend”
- 必须先有一层 contract alignment / decomposition，把 `v16-base` 的调度能力转换成 `v18-base` 所要求的 scheduler contract

建议文件范围：

- 新增或重构以下文件：
  - `vllm_omni/diffusion/sched/step_level_request_scheduler.py`
  - `vllm_omni/diffusion/sched/request_selection_policy.py`
  - `vllm_omni/diffusion/sched/interface.py`
  - `vllm_omni/diffusion/sched/base_scheduler.py`
- 若 merge 过渡期确实需要兼容层，可临时新增：
  - `vllm_omni/diffusion/sched/step_level_request_scheduler_adapter.py`
- 但最终落地物不应以 `Stage1SchedulerAdapter` 命名
- 对应测试：
  - 新增 policy-driven scheduler 单测
  - 补 scheduler contract 行为测试

本阶段应完成的事情：

- 明确最终接入 `v18-base` 的类应叫 `StepLevelRequestScheduler`，并且其实现接口必须是 `SchedulerInterface`
- 明确以下映射关系：
  - `OmniDiffusionRequest.request_ids` 与 `sched_req_id` 的关系
  - waiting / running / preempted / finished 的状态迁移
  - `schedule()` 输出与 executor 输入的契约
  - `update_from_output()` 如何消费 chunk 执行结果
  - `abort/preempt/finish` 的优先级和幂等语义
- 把历史 `Stage1Scheduler` 的能力拆分成三层：
  - `RequestScheduler` 宿主生命周期状态机
  - `RequestSelectionPolicy` 排队 / 选择策略层
  - chunk lifecycle / execution controller
- 明确谁是单一事实来源：
  - request 对象上的生命周期字段
  - scheduler-owned state
  - executor / worker 内部 context
- 形成一个可以被 `DiffusionEngine` 透明选择的 backend 实现，而不是让 engine 理解两套 scheduler 语义

推荐产物：

- 一个 `StepLevelRequestScheduler(SchedulerInterface)`，必要时以 `RequestScheduler` 为宿主基类或直接复用其状态机语义
- 一组 `RequestSelectionPolicy` 实现，用来承载 `fcfs / p95-first / p95-fusion` 等策略
- 如果过渡期必须存在兼容层，优先叫 `StepLevelRequestSchedulerAdapter`
- 一份状态迁移表
- 一份错误语义表

建议验证：

```bash
pytest -q tests/diffusion/test_step_level_request_scheduler.py
pytest -q tests/diffusion/test_runtime_profile_estimator.py
```

并补至少一组新的 contract / policy 测试：

```bash
pytest -q tests/diffusion/test_request_selection_policy.py
```

完成标准：

- 历史 `Stage1Scheduler` 的能力已经被拆解并重组为符合 `v18-base` 逻辑的 `StepLevelRequestScheduler`
- `DiffusionEngine` 后续只需要“选择 backend”，不需要理解旧 scheduler 的内部通信模型
- Phase 4 不再同时承担协议迁移和控制流重构两件大事

### Phase 3: 先移植 pipeline / model runner 的 step API

目标：

- 把“单个 diffusion request 如何 prepare / step / finalize”先做成稳定 API
- 这一层是后续 engine 真正接入 step-chunk 的前提

建议文件范围：

- `vllm_omni/diffusion/worker/diffusion_model_runner.py`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_layered.py`
- `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py`
- `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
- `tests/diffusion/test_diffusion_model_runner.py`
- `tests/diffusion/test_qwen_image_step_chunk.py`

本阶段应完成的事情：

- 基于 `v18-base` 的 pipeline helper 结构，植入 `prepare_generation / step_generation / finalize_generation`
- 让 `DiffusionModelRunner` 支持 context 生命周期
- 把 `v18-base` 已有的观测性改动也保留下来，例如 peak memory / PERF timing
- 先保证“step API 存在且正确”，不要求 engine 已经用它们调度

关键策略：

- `pipeline_qwen_image.py` 以 `v18-base` 的 helper 分层为基底改
- `pipeline_wan2_2.py` 以 `v18-base` 的 PERF instrumentation 为基底改
- 不要简单回退到 `v16-base` 的旧 pipeline 结构

建议验证：

```bash
pytest -q tests/diffusion/test_diffusion_model_runner.py
pytest -q tests/diffusion/test_qwen_image_step_chunk.py
```

完成标准：

- pipeline 层已具备可分步执行能力
- forward 仍能在 step-chunk 关闭时保持 `v18-base` 风格行为
- 新旧路径可以并存，不强制所有请求进入 step API

### Phase 4: 再把 engine / executor 接入 scheduler loop

目标：

- 在前面四层都稳定后，才真正改 `engine` 和 `executor`
- 这是最危险的一轮，要尽量减少同时修改的面

建议文件范围：

- `vllm_omni/diffusion/executor/multiproc_executor.py`
- `vllm_omni/diffusion/diffusion_engine.py`
- `vllm_omni/entrypoints/async_omni_diffusion.py`
- `tests/diffusion/test_diffusion_engine_stage1.py`
- 可能还要补或更新和 engine 行为相关的测试

本阶段应完成的事情：

- 决定 scheduler 的拥有者关系
  - 更推荐保留 `v18-base` executor 结构，再把 `StepLevelRequestScheduler` 接进去
  - 不建议整体回退成 `v16-base` 的 executor 架构
- 决定 scheduler 选择机制
  - 更推荐“旧路径 / step-level request scheduler 路径”双分支共存
  - 默认先保守地维持旧路径，逐步把新路径放到显式配置后
- 基于 Phase 2.5 的 adapter 结果接入 scheduler backend
  - `DiffusionEngine` 只负责选择 `RequestScheduler` 或 `StepLevelRequestScheduler`
  - 不直接吸收 `v16-base` `Stage1Scheduler` 的内部执行语义
- 把 `diffusion_engine.py` 改成 chunk-aware loop
- 让 `request.max_steps_this_turn`、`executed_steps`、`scheduler_chunk_budget_steps` 真正参与运行时控制
- 保留 `v18-base` 的 profiling / observability 入口

建议验证：

```bash
pytest -q tests/diffusion/test_diffusion_engine_stage1.py
pytest -q tests/diffusion/test_diffusion_model_runner.py
pytest -q tests/diffusion/test_step_level_request_scheduler.py
```

完成标准：

- 单实例调度闭环真正打通
- step chunk 开启时能工作，关闭时仍尽量退化到原 `v18-base` 行为
- 出现调度异常时，理论上仍可回退到非 step-chunk 模式
- `StepLevelRequestScheduler` 接入后，旧 scheduler 执行路径仍可通过配置保留和验证

### Phase 5: 最后处理 serving metadata 和 benchmark 对接

目标：

- 当前四轮完成后，内核已经打通
- 这时再把 request metadata 和 benchmark 入口层接上，风险最低

建议文件范围：

- `vllm_omni/entrypoints/openai/api_server.py`
- `vllm_omni/entrypoints/openai/serving_chat.py`
- `vllm_omni/entrypoints/openai/serving_video.py`
- `vllm_omni/entrypoints/openai/protocol/images.py`
- `vllm_omni/entrypoints/openai/protocol/videos.py`
- `benchmarks/diffusion/backends.py`
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
- `tests/entrypoints/test_async_omni_diffusion.py`

本阶段应完成的事情：

- 把 `slo_ms` / `deadline_ts` / `estimated_cost_s` 这种 scheduler-facing metadata 从请求入口打通
- 统一 request-id 策略
- benchmark 侧改成兼容 `v18-base` 的视频异步 API 语义

建议验证：

```bash
pytest -q tests/entrypoints/test_async_omni_diffusion.py
pytest -q tests/entrypoints/test_async_omni_diffusion_config.py
```

如果有可运行的本地 smoke test，再额外做：

```bash
python -m vllm_omni.entrypoints.openai.api_server ...
```

完成标准：

- step chunk / 实例内调度已经能从 OpenAI entrypoint 收到元数据
- benchmark 工具链可用于回归
- 普通不带调度元数据的请求保持旧语义

### Phase 6: global scheduler 最后单独并

目标：

- 在 step chunk / 实例内调度已经稳定后，再单独并 `global scheduler`
- 因为它低冲突、耦合弱，完全没必要提前混入核心 merge

建议文件范围：

- `vllm_omni/global_scheduler/**`
- `profile/**`
- `z_configs/**`
- `benchmarks/diffusion/scripts/global_instance_scheduler_v2/**`
- 相关 README / benchmark 脚本

建议验证：

- 启动 global scheduler smoke test
- 跑最小 benchmark case

## 高风险文件的合并策略

### `vllm_omni/diffusion/diffusion_engine.py`

- 不建议直接拿 `v16-base` 覆盖
- 推荐做法：
  - 保留 `v18-base` 主体结构
  - 手工植入 chunk-aware loop
  - 把 scheduler metrics 聚合逻辑一点点接进去
  - 明确保留一个 `step chunk disabled -> old path` 的控制分支

### `vllm_omni/diffusion/executor/multiproc_executor.py`

- 不建议完全回退到 `v16-base` 的 scheduler ownership 设计
- 推荐先保留 `v18-base` broadcast / worker 启动结构，再嵌入 `StepLevelRequestScheduler`
- 推荐增加明确的 scheduler 选择点，而不是把旧路径直接删掉
- 如果 policy-driven scheduler 需要特殊执行语义，优先通过 adapter / 明确 hook 暴露，不要让 executor 同时兼容两套隐式 contract

### `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`

- 明确以 `v18-base` helper 分层为基底
- 把 `prepare_generation / step_generation / finalize_generation` 重新挂进去
- `forward` 里保留旧路径和新路径双分支

### `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`

- 明确以 `v18-base` PERF timing / progress 结构为基底
- 把 step-chunk 生命周期植入，而不是反过来
- PERF instrumentation 和旧 forward 语义都不要因为 step chunk 接入而丢失

## 每阶段都要做的兼容性检查

建议每轮至少确认两件事：

1. `step chunk` 关闭时，旧路径仍可运行
2. `step chunk` 开启时，新路径能进入对应逻辑

到新 scheduler backend 集成阶段，再额外确认：

3. 旧 scheduler 路径仍可运行
4. `StepLevelRequestScheduler` 路径只在显式配置下启用
5. `StepLevelRequestScheduler` 满足 `SchedulerInterface` 契约，不要求 engine 额外分支理解旧语义

如果测试不全，至少在代码层保证：

- 默认配置不自动开启 step chunk
- 所有新逻辑都挂在显式开关后面
- 无调度元数据请求不会被强制套上 deadline / cost 语义
- `StepLevelRequestScheduler` 不是默认唯一入口，旧 scheduler 仍有保留开关
- scheduler backend 的多样性应被收敛在 contract adapter 层，而不是扩散到 engine 主循环

## 我建议你真正开始 merge 时的第一刀

第一轮不要碰 `engine`、`executor`、`pipeline` 大文件。

先做这组最小闭环：

1. `vllm_omni/diffusion/context.py`
2. `vllm_omni/diffusion/request.py`
3. `vllm_omni/diffusion/data.py`
4. `vllm_omni/entrypoints/async_omni.py`
5. `vllm_omni/entrypoints/cli/serve.py`
6. `tests/diffusion/test_diffusion_request.py`
7. `tests/entrypoints/test_async_omni_diffusion_config.py`

原因：

- 这组文件先把“数据模型”和“配置入口”立住
- 一旦这组稳定，后面 scheduler / pipeline / engine 的改动才有清晰宿主
- 这也是最适合先提交的第一批 commit

## 一句话执行策略

- 先立 request/context/config
- 再立 scheduler primitive
- 再做 pipeline step API
- 再做 engine / executor 真正接线
- 最后才做 serving / benchmark / global scheduler
