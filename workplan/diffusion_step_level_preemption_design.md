# Diffusion Step-Level 调度与抢占改造设计

## 1. 背景

当前 diffusion 在线主路径仍然是 request-mode：

- `DiffusionEngine.step()`
- `DiffusionEngine.add_req_and_wait_for_response()`
- `scheduler.schedule()`
- `executor.add_req(req)`
- worker `generate()`
- `DiffusionModelRunner.execute_model()`
- `pipeline.forward(req)`

这条路径的核心特征是：

- 调度粒度是“整请求”，不是“单 denoise step”
- scheduler 当前只决定“下一个谁开始跑”
- 请求一旦开始执行，就会在 `pipeline.forward()` 中一次跑到底
- `_rpc_lock` 当前跨整次请求持有，新的请求无法在当前请求运行时进入调度面
- `abort()` 尚未打通，`preempt_request()` 也没有形成端到端闭环

因此，当前系统并不具备真正可用的 step-level 调度与抢占能力。

## 2. 目标

本方案要完成的目标是：

1. 将 diffusion 在线执行从 request-level run-to-completion 改为 step-level 调度。
2. 支持在 step 边界进行 cooperative preempt。
3. 保持当前对上层的同步接口语义，即单次 `step()` 调用仍然在请求完成后返回。
4. 保持 request-mode 回滚路径，避免一次性替换全部执行模式。

## 3. 非目标

本轮明确不做：

- step 内中断
- kernel 级强制抢占
- `batch_size > 1` 的 step-level continuous batching
- step mode 下的 LoRA 支持
- step mode 下的 `cache_backend`
- step mode 下的 KV transfer 对齐
- 所有 diffusion pipeline 一次性适配

本轮抢占的定义是：

- 只在 step 边界生效
- 当前 step 执行完成后，运行中的请求可以失去下一轮执行权
- runner 侧保留 request state，下次被重新调度时恢复

## 4. 当前代码现实

### 4.1 已有可复用能力

- `RequestScheduler.schedule()` 已区分：
  - `scheduled_new_reqs`
  - `scheduled_cached_reqs`
- `_BaseScheduler.preempt_request()` 已具备将运行中请求移回 waiting 队列的基础行为
- worker 侧已经具备 stepwise 执行接口：
  - `DiffusionWorker.execute_stepwise()`
  - `DiffusionModelRunner.execute_stepwise()`
- `DiffusionModelRunner.state_cache` 已经能持有 request 级 stepwise 状态
- `RunnerOutput` 已能表达：
  - `req_id`
  - `step_index`
  - `finished`
  - `result`
- `MultiprocDiffusionExecutor.collective_rpc()` 已能把 `execute_stepwise` 发往 worker

### 4.2 当前缺口

- engine 主循环仍然只消费 request-mode 输出
- `RequestScheduler.update_from_output()` 仍然把一次输出视为 terminal
- 当前 `add_req_and_wait_for_response()` 是“调用线程自己驱动调度循环”，不适合多请求交错
- 非目标请求的中间或最终结果没有统一存储位置
- `abort()` 未打通到 scheduler 和 waiter
- stage client 当前只有 `task.cancel()`，没有真正下钻到 engine abort

## 5. 设计结论

### 5.1 结论摘要

- 方案评级：有条件可实施
- 总体风险：中
- 成熟度：可进入实现
- 推荐决策：先做 `step_execution=True` 场景下的最小闭环，范围限制在支持 stepwise contract 的 pipeline 上

### 5.2 核心原则

1. 抢占只发生在 step 边界。
2. scheduler 仍然拥有 request 生命周期。
3. worker/runner 仍然拥有 stepwise 执行状态。
4. engine 不再让每个调用线程各自跑调度循环，而是使用中心化 step loop。
5. request-mode 路径保留，step-mode 通过配置显式启用。

## 6. MVP 架构

### 6.1 新的执行模型

对开启 `step_execution=True` 的 diffusion engine，执行模型改为：

1. 请求进入 scheduler。
2. engine 后台 loop 每次只调度一个 step。
3. worker 执行一次 `execute_stepwise()`。
4. 如果请求未完成，则保留 runner state 并回到 scheduler。
5. 如果请求完成，则保存最终 `DiffusionOutput` 并唤醒对应请求的等待方。

### 6.2 状态机

MVP 仍然复用现有状态定义：

- `WAITING`
- `RUNNING`
- `PREEMPTED`
- `FINISHED_COMPLETED`
- `FINISHED_ABORTED`
- `FINISHED_ERROR`

step-level 下的关键语义变化如下：

- `RUNNING` 表示“该请求拥有可恢复的 runner state，且当前应继续执行”
- `PREEMPTED` 表示“该请求在 step 边界被换出，runner state 仍然有效”
- 非 finished 的 step 输出不改变 terminal 状态，只推进请求执行进度

## 7. 详细设计

### 7.1 Scheduler 设计

新增一个独立的 step-level scheduler，建议命名：

- `StepLevelScheduler`

建议新增文件：

- `vllm_omni/diffusion/sched/step_scheduler.py`

设计理由：

- 避免把 request-mode 和 step-mode 语义混进同一个 `RequestScheduler`
- 保持回滚简单
- 保持 engine 的默认行为不变

#### 7.1.1 继承关系

建议：

- `StepLevelScheduler(RequestScheduler)`

复用内容：

- `add_request()`
- `waiting/running/finished` 队列
- `sched_req_id` 管理
- `finish_requests()`
- `preempt_request()`

需要覆盖的行为：

- `schedule()`
- `update_from_output()`

#### 7.1.2 schedule() 语义

MVP 采用最简单的 RR 语义：

- 如果当前存在 running 请求，且 waiting 非空，则在本轮调度前先对 running 请求执行一次 `preempt_request()`
- 然后走正常的 `schedule()`
- 因为当前 `batch_size=1`，一次只会产出一个待执行请求

这样可以实现：

- 每个请求每轮最多执行一个 step
- 一旦有新请求进入 waiting，当前长请求会在下一步边界被换出

这就是 MVP 的抢占语义。

#### 7.1.3 update_from_output() 语义

step-level 下，`update_from_output()` 的输入应扩展为 `RunnerOutput`。

语义如下：

- `finished=False`
  - 请求保持 `RUNNING`
  - 不进入 terminal
  - 返回空 finished 集合
- `finished=True`
  - 若 `result.error is None`，进入 `FINISHED_COMPLETED`
  - 若 `result.error is not None`，进入 `FINISHED_ERROR`
- 若请求在 step 执行期间被标记 abort
  - 当前 step 返回后收敛到 `FINISHED_ABORTED`

建议将接口放宽为：

```python
def update_from_output(
    self,
    sched_output: DiffusionSchedulerOutput,
    output: DiffusionOutput | RunnerOutput,
) -> set[str]:
    ...
```

也可以在 `StepLevelScheduler` 中单独收窄为 `RunnerOutput`，但接口层仍建议统一放宽，减少 engine 分支判断复杂度。

### 7.2 Engine 设计

#### 7.2.1 为什么不能沿用当前 per-request loop

当前 `add_req_and_wait_for_response()` 的模型是：

- 谁提交请求，谁驱动调度循环
- 谁的目标请求先结束，谁就返回

这个模型在 request-mode 下可行，但在 step-level 交错执行下会有问题：

- 线程 A 可能推进线程 B 的请求
- 线程 A 不会保存线程 B 的最终输出
- 多个调用线程都尝试 drive scheduler，会让“结果归属”和“等待/唤醒”边界变混乱

因此，step-level 模式下必须切换为：

- 调用线程只负责提交请求并等待结果
- engine 内部单独持有一个中心化 step loop

#### 7.2.2 新增中心化 step loop

在 `DiffusionEngine` 中新增：

- 后台线程：`_scheduler_thread`
- 请求完成通知：`_request_events`
- 最终结果表：`_request_outputs`
- 可选错误表：`_request_errors`

推荐最小实现方式：

- `add_req_and_wait_for_response()`：
  - 注册 request
  - 创建 event
  - 唤醒后台 loop
  - 阻塞等待 event
- 后台 loop：
  - `sched_output = scheduler.schedule()`
  - 若为空且没有请求，等待条件变量
  - 否则执行一步 stepwise RPC
  - 用 scheduler 更新状态
  - 若某请求 finished，则保存其最终输出并唤醒等待方

#### 7.2.3 锁语义

MVP 不引入额外复杂锁层次，保留一个 engine 内部互斥锁即可，但其持有范围必须从“整请求”收窄到“单次 step 调度/RPC”。

也就是说：

- 当前 `_rpc_lock` 不能再包住整次 `add_req_and_wait_for_response()`
- 它只能保护：
  - scheduler 状态修改
  - 单次 worker RPC
  - 单次结果回写

这样才能让：

- 新请求在两个 step 之间进入 waiting 队列
- abort/preempt 请求在两个 step 之间生效

#### 7.2.4 Worker 调用路径

step-level 模式下，engine 不再调用：

- `executor.add_req(req)`

而是调用：

```python
executor.collective_rpc(
    method="execute_stepwise",
    args=(sched_output,),
    unique_reply_rank=0,
)
```

这条路径复用现有能力，无需新增 executor 专用接口。

#### 7.2.5 返回语义

对上层保持不变：

- `DiffusionEngine.step()` 仍然在该请求完整结束后返回

内部变化只是：

- 返回值不再来自一次 `pipeline.forward()`
- 返回值来自该请求最后一次 `RunnerOutput.result`

### 7.3 Abort 与 Preempt 设计

#### 7.3.1 Abort 语义

`DiffusionEngine.abort(request_id)` 在 MVP 中需要真正打通：

- waiting/preempted 请求：
  - 直接标记 `FINISHED_ABORTED`
  - 立即唤醒 waiter
- running 请求：
  - 标记 abort pending
  - 当前 step 返回后收敛为 `FINISHED_ABORTED`

这意味着 abort 延迟上界是：

- 一个 denoise step

#### 7.3.2 Preempt 语义

MVP 不要求先暴露显式 `preempt(request_id)` API。

MVP 的抢占来自调度策略本身：

- 只要 waiting 队列非空，running 请求在下一轮前被换出

如果后续需要显式 preempt API，可在 engine 中增加：

- `preempt(request_id)`

并在下一轮调度前调用 scheduler 的 `preempt_request()`。

### 7.4 Stage/Async 路径闭环

当前 `AsyncOmniDiffusion.abort()` 已会调用 `engine.abort()`。

但 `StageDiffusionClient.abort_requests_async()` 目前只是：

- 取消本地 task

这会带来一个问题：

- 上层任务取消了，但 engine 内部请求仍可能继续跑

因此，MVP 需要补齐：

- `StageDiffusionClient.abort_requests_async()` 在取消 task 的同时，调用底层 engine abort

这样 stage/orchestrator 路径才是闭环的。

## 8. 实现边界

### 8.1 MVP 只支持 stepwise pipeline

MVP 仅在下面条件成立时启用：

- `od_config.step_execution=True`
- pipeline 实现 `SupportsStepExecution`

不满足条件时：

- 继续走原来的 request-mode

### 8.2 MVP 不做多请求 step batching

当前 `_max_batch_size=1` 保持不变。

原因：

- 现有 engine、scheduler output、worker state cache 都是单请求语义优先
- 先验证单请求 step-level 抢占闭环，再做 batch-level continuous batching 风险更低

## 9. 最小改动闭环

建议按下面路径落地。

### 9.1 必改文件

- `vllm_omni/diffusion/diffusion_engine.py`
- `vllm_omni/diffusion/sched/interface.py`
- `vllm_omni/diffusion/sched/__init__.py`
- `vllm_omni/diffusion/stage_diffusion_client.py`
- 新增 `vllm_omni/diffusion/sched/step_scheduler.py`

### 9.2 可选小改文件

- `vllm_omni/diffusion/data.py`

若希望配置更显式，可新增：

- `step_preemption_quantum_steps`
- `enable_step_preemption`

但对 MVP 来说，也可以先不加配置，默认在 `step_execution=True` 时启用 `1-step RR`。

### 9.3 不建议在 MVP 修改的文件

- `vllm_omni/diffusion/executor/multiproc_executor.py`
- `vllm_omni/diffusion/worker/diffusion_worker.py`
- `vllm_omni/diffusion/worker/diffusion_model_runner.py`

原因：

- 这些文件已经具备 MVP 所需能力
- 再改会扩大风险面

## 10. 伪代码

### 10.1 Engine 提交路径

```python
def add_req_and_wait_for_response(self, request):
    with self._rpc_lock:
        sched_req_id = self.scheduler.add_request(request)
        self._request_events[sched_req_id] = threading.Event()
        self._wake_scheduler_loop()

    self._request_events[sched_req_id].wait()

    with self._rpc_lock:
        output = self._request_outputs.pop(sched_req_id)
        self.scheduler.pop_request_state(sched_req_id)
        self._request_events.pop(sched_req_id, None)
        return output
```

### 10.2 Engine 后台 step loop

```python
def _scheduler_loop(self):
    while not self._closed:
        with self._rpc_lock:
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                self._wait_for_new_work()
                continue

        runner_output = self.executor.collective_rpc(
            method="execute_stepwise",
            args=(sched_output,),
            unique_reply_rank=0,
        )

        with self._rpc_lock:
            finished = self.scheduler.update_from_output(sched_output, runner_output)
            if runner_output.finished:
                self._request_outputs[runner_output.req_id] = runner_output.result
            for req_id in finished:
                self._request_events[req_id].set()
```

### 10.3 StepLevelScheduler RR 语义

```python
def schedule(self):
    if self._running and self._waiting:
        self.preempt_request(self._running[0])
    return super().schedule()
```

## 11. 测试与验收

### 11.1 必须新增/更新的测试

- `tests/diffusion/test_diffusion_scheduler.py`
  - step scheduler 未完成请求不会进入 terminal
  - waiting 非空时 running 请求会在 step 边界被 preempt
- `tests/diffusion/test_diffusion_step_pipeline.py`
  - cached request 能从 state cache 恢复
  - stepwise finished 时返回最终 output
- 新增 engine 集成测试
  - 两个请求交错执行时都能拿到正确结果
  - 短请求可以在长请求中途插队完成
  - running request abort 能在一个 step 内收敛
- `tests/entrypoints` 或 stage client 测试
  - stage abort 不再只是 cancel task，而会下钻 engine abort

### 11.2 MVP 验收标准

MVP 视为完成，需要同时满足：

1. `step_execution=True` 的支持 pipeline 可以通过 stepwise 路径完成请求。
2. 两个不同请求可以交错执行，而不是先后整请求串行。
3. 长请求运行中，新短请求到来后，最迟在一个 step 后获得执行机会。
4. running request 的 abort 最迟在一个 step 后生效。
5. request-mode 默认路径无行为回归。

## 12. 风险与回滚

### 12.1 主要风险

- stepwise pipeline 支持面仍然有限
- 每 step 一次 RPC 会增加调度与 IPC 开销
- `1-step RR` 可能过于激进，吞吐可能下降
- stage 路径如果只 cancel task 不 abort engine，会出现悬挂请求

### 12.2 回滚路径

保持下面两个回滚开关即可：

- `step_execution=False`
- 默认 scheduler 仍使用原 `RequestScheduler`

只要新路径不替换默认 request-mode，就可以做到快速回滚。

## 13. 后续改进路线

### 13.1 第一阶段后立即可做

- 显式 `preempt(request_id)` API
- 可配置 `quantum_steps`
- 每请求 preempt 次数、等待时间、step 进度指标

### 13.2 第二阶段

- `batch_size > 1` 的 step-level continuous batching
- remaining-cost / aging / SLA aware 调度策略
- 更细粒度的 fairness 策略

### 13.3 第三阶段

- step mode 下的 `cache_backend`
- step mode 下的 KV transfer
- step mode 下的 LoRA
- 更多 pipeline 的 stepwise 支持

## 14. 建议 PR 切分

### PR1

- 新增 `StepLevelScheduler`
- scheduler unit tests

### PR2

- `DiffusionEngine` 中心化 step loop
- engine 集成测试

### PR3

- 打通 `abort()`
- stage client 控制面闭环

### PR4

- `quantum` 配置
- 指标与观测

## 15. 最终建议

不要把这次改造定义成“立即支持完整抢占式 diffusion serving”。

更准确的工程目标应当是：

- 先把 diffusion execution model 从 request-level 切成 step-level
- 再把抢占定义为 step boundary 上的 cooperative preempt
- 先用 `batch_size=1` 验证语义与恢复正确性
- 最后再进入真正的 policy 和 batching 优化

这是当前代码现实下风险最低、最容易形成闭环的实现路径。
