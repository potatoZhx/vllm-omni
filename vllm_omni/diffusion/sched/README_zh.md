# Diffusion Scheduler 包说明

这个目录包含 diffusion engine 使用的本地 diffusion scheduler 实现。

## 目录文件

- `interface.py`：scheduler 对外契约，以及 scheduler 持有的共享状态类型。
- `base_scheduler.py`：具体 scheduler 复用的队列和状态管理逻辑。
- `request_scheduler.py`：原始 request-level 执行路径使用的 scheduler。
- `step_level_request_scheduler.py`：MVP stepwise 执行路径使用的
  step-level scheduler。
- `policy.py`：step-level scheduler 的等待队列选择策略。

## Step-Level Scheduler 概览

`StepLevelRequestScheduler` 对应的后端配置是
`diffusion_scheduler_backend="step_level_request_scheduler"`。

当前 MVP 的能力边界比较收敛：

- 只支持单 prompt 请求
- scheduler batch size 固定为 `1`
- 每次 dispatch 只执行 `1` 个 diffusion step
- config 校验当前允许：
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`
- LoRA、cache backend、request batching 还不在当前 step-level 路径内

运行流程如下：

1. `add_request()` 创建 scheduler 自己管理的 request state，并放入
   waiting 队列。
2. `schedule()` 从 waiting 队列里选出一个请求，标记为 running，并输出：
   - 新请求第一次调度时放进 `scheduled_new_reqs`
   - 已缓存请求恢复执行时放进 `scheduled_cached_reqs`
3. executor 执行一次 `execute_stepwise()`。
4. `update_from_output()` 消费 `RunnerOutput`，然后做以下三类收敛之一：
   - 请求完成并结束
   - 在 step 边界上 abort
   - 标记成 `PREEMPTED`，重新放回 waiting 队列

当前 step-level scheduler 只负责决定“下一个选哪个 waiting 请求”。它还不负责：

- 多请求 batch 组装
- 可变 chunk 大小分配
- 多请求并行打包

这些能力在当前 MVP 里都是固定的。

## 策略接口

step-level 策略需要实现 `policy.py` 中的 `RequestSelectionPolicy`：

```python
class RequestSelectionPolicy(Protocol):
    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        ...
```

输入含义：

- `waiting`：当前 waiting deque 中的 scheduler request id 顺序
- `request_states`：请求级元数据，例如原始请求和当前状态
- `execution_states`：执行级元数据，例如 arrival time、已执行 step 数、
  dispatch epoch、estimated runtime、abort 标记等

输出含义：

- 返回一个重新排序后的 scheduler request id 列表

当前基类额外提供了一组可选 lifecycle hook：

- `initialize()`
- `on_request_arrival()`
- `on_request_scheduled()`
- `on_step_complete()`
- `on_request_finished()`

这组 hook 的作用是：让迁移过来的策略把少量在线学习状态保留在 policy 对象自己内部，
而不是再次把旧 `Stage1Scheduler` 的算法状态塞回 scheduler 主体。

## 已迁移策略说明

### `fcfs`

严格保持 waiting 队列原始顺序。它仍然是最稳妥的默认基线。

### `sjf`

按“预计剩余运行时间”给 waiting 请求排序。

估时优先级如下：

1. 请求显式携带的 `sampling_params.extra_args["estimated_cost_s"]`
2. `instance_runtime_profile_path` + `instance_runtime_profile_name`
3. scheduler 本地启发式 fallback

对于已经执行过的请求，剩余 cost 会按
`(total_steps - executed_steps) / total_steps` 缩放。

### `sjf_aging`

在 `sjf` 的剩余 cost 基础上，沿用 `v16-base` 的 aged-cost 排序语义：

- 等待越久，请求获得越强的 aging 折扣
- cost 越大，aging 权重会按有界 cost-aware 规则放大

这样既保留了 SJF 对短请求的吞吐偏好，也避免纯 SJF 的明显饥饿问题。

### `sjf_aging_guarded`

在 `sjf_aging` 基础上增加 protected 队列。

当请求等待时间超过以下两者中的较大值时，请求会进入 protected：

- 从已完成请求等待历史中学到的 wait guard
- `2.0 * estimated_remaining_cost_s`

进入 protected 后，请求会整体排在 normal 请求之前，且 protected 组内按到达时间排序。

别名：

- `sjf_aging_guard` 会被接受并归一化为 `sjf_aging_guarded`

### `sjf_aging_guarded_tail`

在 `sjf_aging_guarded` 基础上，迁入了 `v16-base` 的 tail-sink 语义：

- 只有 protected 且 super-heavy 的请求才可能被 sink
- 只允许严格的 5% 全局 / 滑动窗口 defer budget
- 每轮 reorder 最多 sink 1 个请求
- 被 sink 的请求在后续 requeue 中会保持尾部状态，直到 hard escape 释放

当前 `v18-base` 最小落地版本的明确限制：

- 这次迁移只覆盖 waiting 队列重排语义
- 还没有把旧实现里的 chunk-budget override 一并迁过来，例如 idle-only `3x`
  chunk 扩张，因为当前 step-level backend 仍然固定为单 step dispatch

### `p95-first`

迁入了 `v16-base` 的 normalized tail-pressure 排序主路径：

- 用真实 step runtime 学习 observed service time
- 用 end-to-end latency / cumulative execute time 学习 slowdown
- 通过 greedy 方式，按归一化 tail pressure 重排 waiting 队列

当前落地刻意保持收敛：

- 保留了 learned service-rate / slowdown 主链路
- 没有把旧版大规模 CLI 调参面一起暴露出来，例如 `base_ms` / `max_ms` /
  backlog alpha / bucket / fusion 等参数

## 如何新增一个 Step-Level 策略

### 1. 实现策略类

简单策略可以直接加在 `policy.py` 中。复杂策略建议拆到同目录下的新文件，
然后在 builder 中引入。

示例：

```python
class ShortestExecutedStepsPolicy:
    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        return sorted(
            waiting,
            key=lambda req_id: execution_states[req_id].executed_steps,
        )
```

实现建议：

- `waiting` 应当视为当前可运行请求的唯一真值来源
- 排序要保持确定性，tie-break 尽量回退到原始 `waiting` 顺序
- 只读取 scheduler 持有的状态，不要直接依赖 worker 内部状态
- 优先把策略自己的在线学习状态保留在 policy 对象内部，并由 lifecycle hook 驱动，
  不要把它们重新塞回 scheduler 主体

### 2. 在 builder 中注册

更新 `build_request_selection_policy()`，让 scheduler 能通过
`instance_scheduler_policy` 构造出新策略。

示例：

```python
def build_request_selection_policy(name: str) -> RequestSelectionPolicy:
    if name == "fcfs":
        return FCFSSelectionPolicy()
    if name == "shortest_executed_steps":
        return ShortestExecutedStepsPolicy()
    raise NotImplementedError(...)
```

### 3. 放开 config 校验

当前 `OmniDiffusionConfig` 对 step-level policy 使用显式 allowlist。
如果要启用新策略，需要同步更新
`vllm_omni/diffusion/data.py` 里的校验逻辑。

最少要做的事情：

- 扩展已有 allowlist
- 继续保留 `diffusion_enable_step_chunk=True` 的要求
- 明确判断新策略在当前 batch size 为 `1` 的 MVP 路径下是否安全

### 4. 只在必要时扩展执行元数据

如果新策略需要更多调度信号，可以扩展 `interface.py` 里的
`DiffusionExecutionState`，并在 scheduler 更新时填充这些字段。

建议优先保持策略输入停留在 scheduler 本地状态层。若确实需要模型执行侧信息，
也应先把它转换成 scheduler 元数据，再让策略消费，而不是让策略直接解析 worker 输出。

### 5. 增加测试

至少要覆盖：

- `tests/diffusion/test_diffusion_scheduler.py`
- `tests/entrypoints/test_async_omni_diffusion_config.py`

建议覆盖的测试点：

- builder 是否能正确构造目标策略
- 排序行为是否符合预期且具有确定性
- config 校验是否只在允许的 backend 组合下接受新策略
- 不支持的组合是否仍然会明确失败

### 6. 复查 serve 接入

当前 serve 路径已经会透传 `--instance-scheduler-policy`，所以多数新策略只需要：

- builder 注册
- config 校验放开

如果新策略还引入了额外调优参数，则需要继续把参数串到：

- `vllm_omni/entrypoints/cli/serve.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/diffusion/data.py`

## 当前 Step-Level 的限制

设计新策略时，需要牢记当前实现边界：

- 没有真正的 request batching，`_max_batch_size` 固定为 `1`
- 策略当前只能重排 waiting 队列，不能分配多 step budget
- 已迁入的 `sjf_aging_guarded_tail` 也还没有迁 chunk-budget override 行为
- 请求恢复依赖 cached scheduler id，而不是重新发送完整请求负载
- abort 发生在 step 边界，不是 mid-step interrupt
- 策略效果上限受 `DiffusionExecutionState` 中已有元数据约束

如果后续引入多请求调度或者可变 chunk budget，这份 README 也需要同步更新，
因为那时策略契约就不再只是“重排 waiting 队列”了。
