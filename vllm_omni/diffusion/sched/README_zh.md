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
- config 校验目前只允许 `instance_scheduler_policy="fcfs"`
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

当前默认实现 `FCFSSelectionPolicy` 只是保留 waiting 队列原始顺序。

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
- 策略应保持纯函数语义，不要在排序过程中修改 scheduler 状态

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

当前 `OmniDiffusionConfig` 会明确拒绝除 `fcfs` 之外的 step-level 策略。
如果要启用新策略，需要同步更新
`vllm_omni/diffusion/data.py` 里的校验逻辑。

最少要做的事情：

- 把现在写死的 `fcfs` 限制改成 allowlist
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
- 请求恢复依赖 cached scheduler id，而不是重新发送完整请求负载
- abort 发生在 step 边界，不是 mid-step interrupt
- 策略效果上限受 `DiffusionExecutionState` 中已有元数据约束

如果后续引入多请求调度或者可变 chunk budget，这份 README 也需要同步更新，
因为那时策略契约就不再只是“重排 waiting 队列”了。
