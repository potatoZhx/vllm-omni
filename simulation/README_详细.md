# 离散事件模拟器设计说明

## 目标

为了在**不实际运行算法、不发起真实压力测试**的前提下估计算法性能，可以构建一个**离散事件模拟器**。

模拟器不执行真实推理，只依赖一张预先测得的执行时间表：

- 请求类型
- worker 配置
- 该请求在该配置下的运行时间

在此基础上，模拟调度器与多个 worker 的行为过程，得到吞吐、延迟、P50、P95、P99、完成请求数、失败请求数、SLO 达成率等指标。指标口径尽量与 benchmark 脚本保持一致；能对齐的尽量对齐，无法严格对齐的指标单独说明。

## 配置入口（合并后的示例）

- `simulation/config/simulation_config.yaml`：主配置（通常用于 random 请求 mix、多 worker、多策略对比）。  
- `simulation/config/simulation_config_tmp.yaml`：单实例 + trace 数据集示例（来自合并的 `simulation_tmp`），配套 `simulation/profile/qwen.json` 与 `simulation/xhf_tmp/datasets/datasetX.trace.txt`。

---

## 核心建模假设

### 1. 忽略通信与调度开销

默认认为：

- 请求到达调度器不耗时
- 调度器把请求放入 worker 队列不耗时
- worker 返回结果不耗时

也就是说，请求一旦被调度，就在同一时刻进入目标 worker 队列。

---

### 2. 服务时间由查表得到

每个请求在某个 worker 上的执行时间，不通过真实运行得到，而是由预先测得的数据直接查表得到。

因此模拟器本质上是在做：

- 请求到达过程模拟
- 队列等待过程模拟
- 调度决策模拟
- 完成事件统计

而不是做真实推理。

---

### 3. 采用固定模拟时长作为终止条件

模拟使用**固定模拟时长**而不是固定请求数作为结束条件。

设总模拟时长为 `T_end`。

规则如下（与 `build_requests` + 主循环一致）：

- 仅当「下一条预生成请求」的 `arrival_time <= T_end` 时，才会在对应时刻执行到达并入队
- 否则调度侧下一时刻为 `inf`，不再入队
- 之后系统继续运行，直到所有 worker 都空闲且队列为空，模拟结束

也就是说：

- `T_end` 控制的是**请求注入阶段**
- 模拟最终结束时刻可能大于 `T_end`
- 到达间隔在列表里为 `1/rps`，等价为均匀到达；实现上 **不** 用「每次 += 1/rps」递推，而用预计算的 `arrival_time` 序列

---

## 系统组成与两层调度（实现：simulation/simulation.py）

当前实现是 **两层决策**，与 `simulation_config.yaml` 里两类字段一一对应：

| 层级 | YAML 配置 | 代码入口 | 回答的问题 |
|------|-----------|----------|------------|
| **全局（路由）** | `scheduler.algorithms_to_run` | `DISPATCH[name]`，如 `dispatch_min_queue_length` | 新到达的请求 **入哪一台 worker 的队列** |
| **实例内（单 worker）** | `scheduler.instance_scheduler_policies` | `_pop_next_from_queue`、chunk 完成分支 | 该 worker **下一个服务谁**、是否 **按 chunk 切分执行** |

一次完整仿真会对 **`algorithms_to_run` × 展开后的 `instance_scheduler_policies`** 做 **笛卡尔乘积**，每个组合单独跑一遍 `run_simulation`，输出中的 `algorithm` 形如 `short_queue_runtime_sjf_preempt_4`。

**与真系统 / benchmark 对齐的主要假设：**

- **没有客户端 `max-concurrency`**：请求在 `build_requests` 里已生成 `arrival_time`，到点即 **dispatch + 入队**，不因服务端忙而推迟注入（开环到达）。
- **每个 worker 同一时刻最多执行一条请求**：`current_request` 占槽；其余在 `queue` 排队。
- **全局调度只在入队时调用一次**：只决定 **目标 worker**；队列内顺序由 **实例内策略** 在派工「取队」时决定；入队 **总是** `append` 队尾。

---

## 请求列表与时间轴（进入主循环之前）

- 条数：`N = ceil(t_end * rps)`（`rps > 0`）。
- 到达时刻：`arrival_time = 0, 1/rps, 2/rps, …`（与 `diffusion_benchmark_serving` 固定时长注入语义对齐）。
- **注入停止**：预生成列表中下一条 `arrival_time > t_end` 时，调度器侧下一事件为 `inf`。已入队请求继续服务，`finish_time` 可以晚于 `t_end`。
- **负载**：`dataset=random` 按 `random_request_config` 的 **weight** 预采样 `N` 条；`trace` 只决定 **类型循环**，间隔仍如上。

随后 `assign_slo` 按 profile 为每条请求填 `slo_ms`（统计 SLO 达成率用）。

---

## Worker 运行态字段

- `queue`：队尾入队；出队位置由实例内策略决定（未必队首）。
- `current_request` / `next_time`：`next_time` 下一次 **完成** 时刻（整段服务 **或** 一个 chunk）；`inf` 且无为空则表示空闲。
- **`short_queue_runtime` / `short_queue_runtime_max_class_balanced`**：每台维护 `_queued_work`（队列内剩余工作量秒数的缓存）；后者另有 `_max_class_count`（「最大类」作业个数，用于路由偏置；**仅在请求完成时递减**，抢占重入队不减）。

抢占路径下请求会带 `_executed_steps` 等；**剩余服务时间**由 `lookup` 整段与已执行步数比例估算（`_remaining_latency_s`）。

---

## 离散事件主循环（调度器 + Worker 工作流）

每轮：`t = min(调度器下一时刻, 所有 worker 的 next_time)`，直至结束。代码中在同一 `t` 上可能 **先处理到达、再处理完成**（或相反取决于 `min` 来源），再 **统一做一轮全 worker 立即派工**。

### A. 调度器事件：到达并入队

当 `t` 等于「尚未注入的下一条」的 `arrival_time` 且 `≤ t_end`：

1. `state["current_time"] = t`（供 `sjf_aging` 等使用）。
2. `wi = dispatch_fn(req, workers, state)`：**全局路由**。
3. `queue.append(req)`。
4. 若为 `short_queue_runtime*`：对该 worker `_queued_work +=` 本条整段查表时间；`max_class_balanced` 时再判断是否最大类并 `_max_class_count += 1`。
5. 调度器下一时刻 := 列表中下一条的 `arrival_time`，若无或 `> t_end` 则为 `inf`。

**本步不派工。**

### B. Worker 完成事件

凡 `next_time == t` 的 worker：

1. 清空 `current_request`，`next_time = inf`。
2. **`sjf_preempt` 且 chunk 后仍有剩余步**：更新 `_executed_steps`，按需修正 `_queued_work`，请求 **回到同一 worker 队尾**（未完成不离系统）。
3. **否则** 记 `finish_time`、`latency`，加入 `completed`；`max_class_balanced` 时对最大类 `_max_class_count -= 1`。

### C. 立即派工（不推进时钟）

凡 `next_time == inf` 且 `queue` 非空：

1. `req = _pop_next_from_queue(..., instance_scheduler_policy)`。
2. `short_queue_runtime*`：从 `_queued_work` 减去本条将取走的量（preempt 用 **剩余**；非 preempt 用入队缓存 `_svc`）。
3. `first_start_time` / `start_times` / `start_time`（`start_time` 为 **首次** 开始时刻，兼容字段）。
4. **服务时长**：`sjf_preempt_*` → `_chunk_duration_s`（步数切分、小请求阈值、可选 image/video budget）；**其它策略** → 整段 `lookup`，**不分块**。
5. `next_time = t + st`，`current_request = req`，`chunk_service_times` 追加 `st`。

### D. 结束条件

调度器下一时刻为 `inf`，且所有 worker `next_time == inf` 且 `queue` 为空。

---

## 全局调度算法（`algorithms_to_run`）

- **`round_robin`**：维护游标；有空闲则只在 **空闲 worker** 上轮询，否则在 **全体** 上轮询（对齐 `RoundRobinPolicy`）。
- **`min_queue_length`**：若存在空闲 worker，**只在空闲集合**里选；若全忙则在 **全体** 上选。评分 `len(queue) + (current_request ? 1 : 0)` — **按条数，不按秒**。
- **`short_queue_runtime`**：在 **全部 worker** 上算 `_total_outstanding_work_s`（队列剩余 + 运行中剩余），取 **最小**。用 `_queued_work` 增量维护。
- **`short_queue_runtime_max_class_balanced`**：非最大类同上；最大类优先 `_max_class_count` 最少，再按 short_queue_runtime 打破平局。

---

## 实例内调度（`instance_scheduler_policies`）

仅在 **取队** 与 **`sjf_preempt` 完成处理** 中生效。**同一次 yaml 里勾选了 `sjf_preempt` 也不会让 `fcfs`/`sjf` 曲线分块**——每条曲线用自己的 `instance_scheduler_policy` 字符串。

| 策略 | 取队 | 执行 |
|------|------|------|
| `fcfs` | 队首 | 整段 `lookup` |
| `sjf` | 整段服务时间最短 | 整段 |
| `sjf_aging` / `sjf_aging_0.15` | `st - factor * (t - arrival)` 最小 | 整段 |
| `sjf_preempt_N` | **剩余**时间最短（SRPT） | chunk，每段至多 `N` 步；末段不足不补齐；可配小请求阈值与 image/video chunk |

YAML 里写 `sjf_preempt` 会展开为多条 `sjf_preempt_4`、`sjf_preempt_8`（`sjf_preempt_chunk_budgets`）。仅这些曲线读取 `sjf_preempt_small_request_threshold_ms` 等 preempt 专用参数。

**指标**：若有 chunk，`service_time = sum(chunk_service_times)`，`waiting_time = latency - service_time`。

---

## 与旧版 README 的差异说明

- 调度器下一时刻 **不是** 事件里 `+= 1/rps`，而是预生成列表的 **下一条 `arrival_time`**；等价均匀到达，但停止条件绑在 `t_end` 与列表上。
- 「派工取请求」**不限于队首**，取决于上表实例内策略。

---

## 指标统计

指标尽量对齐 benchmark 脚本；能支持的尽量支持。

### 基础时间字段（与当前 simulation.py 输出一致）

- `arrival_time`：到达并入队时刻（dispatch 时刻）。
- `first_start_time` / `start_time`：**第一次**拿到 worker 的时刻（多次 chunk 间可能等待，`start_time` 为兼容字段，与 `first_start_time` 相同）。
- `start_times`：各段执行（每次派工）的开始时刻列表；`sjf_preempt` 下长度可大于 1。
- `finish_time`：最后一次片段完成时刻。

由此：

- `latency = finish_time - arrival_time`（端到端）。
- **非抢占或单段**：`service_time` 可用 `finish_time - start_time`；**有 chunk 时**：`service_time = sum(chunk_service_times)`（各段 profile 按比例切分之和）。
- `waiting_time = latency - service_time`（排队 + 被抢占等待，不含真实网络）。

以上均不含网络与真实调度器开销。

---

### 建议支持的指标

#### 1. 请求完成情况

- `completed_requests`
- `failed_requests`

如果当前模型中没有失败机制，则：

- `failed_requests = 0`

但建议接口层预留失败统计能力，便于以后扩展超时丢弃、队列溢出等机制。

---

#### 2. 吞吐指标

- `duration`
- `throughput_qps`

这里建议定义：

- `duration = 最后一个完成请求的完成时间 - 第一个请求到达时间`

若请求从 `t=0` 开始到达，也可直接取：

- `duration = 最后完成时刻`

然后：

- `throughput_qps = completed_requests / duration`

这与 benchmark 中“完成请求数 / 总持续时间”的口径保持一致。

---

#### 3. 延迟指标

基于所有完成请求的 `latency` 统计：

- `latency_mean`
- `latency_median`
- `latency_p50`
- `latency_p95`
- `latency_p99`

说明：

- benchmark 至少常用 `mean / median / p99`
- 如果实现方便，建议一并支持 `p50 / p95 / p99`

---

#### 4. 等待与服务分解指标

这类指标 benchmark 通常未必直接输出，但模拟器很适合支持：

- `waiting_time_mean`
- `waiting_time_p95`
- `waiting_time_p99`
- `service_time_mean`
- `service_time_p95`
- `service_time_p99`

这些指标有助于区分：

- 调度/排队造成的慢
- 执行时间本身造成的慢

---

#### 5. SLO 指标

若启用 SLO，则每个请求可以带一个 `slo_ms`。

定义：

- 若 `latency * 1000 <= slo_ms`，则该请求满足 SLO

可统计：

- `slo_defined_total`
- `slo_met_success`
- `slo_attainment_rate`

其中：

- `slo_attainment_rate = slo_met_success / slo_defined_total`

这与 benchmark 的语义基本一致。

---

#### 6. 队列与资源利用指标（推荐额外支持）

模拟器还很适合支持 benchmark 不容易直接给出的指标：

- 每个 worker 的忙碌时间
- 每个 worker 的利用率
- 每个 worker 的平均队列长度
- 全局平均队列长度
- 最大队列长度
- 请求分配分布

例如：

- `worker_utilization = busy_time / total_simulation_time`

这些指标对分析调度算法非常有帮助，建议支持。

---

## 与 benchmark 口径的关系

为了便于比较，建议模拟器输出时尽量采用 benchmark 风格字段名，例如：

- `duration`
- `completed_requests`
- `failed_requests`
- `throughput_qps`
- `latency_mean`
- `latency_median`
- `latency_p50`
- `latency_p95`
- `latency_p99`
- `slo_attainment_rate`

这样可以更方便地与现有作图和结果分析脚本对接。

需要注意的是，模拟器的 `latency` 不包含网络、序列化、真实调度开销，因此数值更偏向“理想化系统表现”。

---

## 建议的实现原则

### 1. 先把事件机制写干净

建议先只实现：

- 调度器请求到达
- worker 请求完成
- 立即派工检查
- 固定模拟时长停止注入

不要一开始就加入复杂失败机制。

---

### 2. 请求执行时间查表要独立封装

建议单独提供一个函数，例如：

- 输入：请求类型、worker 配置
- 输出：执行时间

这样后续如果改成：

- 线性回归估计
- 插值估计
- 加噪声扰动

都比较方便。

---

### 3. 统计模块独立

建议把事件模拟与指标统计分开：

- 模拟器负责产生日志或请求结果记录
- 统计模块负责计算 mean / p99 / throughput / SLO 等指标

这样结构更清晰，也更容易验证正确性。

---

## 最终总结

该模拟器是一个 **两层离散事件调度模拟器**：

- **全局**：按 `algorithms_to_run` 在请求到达时选择 worker 并入队（开环到达，无客户端并发上限）。
- **实例内**：按 `instance_scheduler_policies` 从队列取任务；仅 `sjf_preempt_N` 将一次服务拆成多个 **chunk 完成事件**，其余策略均为 **整段查表一次完成事件**。
- **时钟推进**：下一事件时刻 = 下一到达或任一 worker 的 `next_time`；到达处理与完成处理在同一轮 `t` 上按代码顺序执行后，再对所有空闲 worker 做 **立即派工**（不耗额外仿真时间）。

终止策略：

- `arrival_time > t_end` 的请求不再注入
- 直至所有 worker 空闲且队列清空

在此框架下可对路由算法 × 实例内策略 × profile × 负载做组合实验，输出与 `diffusion_bench/plot_results` 对齐的 JSON/CSV 指标。