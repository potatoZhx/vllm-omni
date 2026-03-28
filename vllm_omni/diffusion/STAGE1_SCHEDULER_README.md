# Stage1 Scheduler README

本文档基于当前仓库实现，说明 `vllm_omni/diffusion/stage1_scheduler.py` 的实际行为、配置入口、排序规则和可观测指标。

适用范围：

- Stage-1 diffusion 请求的实例内调度
- `vllm_omni/diffusion/stage1_scheduler.py`
- 当前仓库里的 CLI、配置校验和单元测试

不适用范围：

- 跨实例全局调度
- 线上 SLO 策略设计
- 历史实验分支中的临时约定

## 1. 当前支持的策略

`OmniDiffusionConfig.instance_scheduler_policy` 当前支持以下 17 种取值：

- `fcfs`
- `sjf`
- `sjf_aging`
- `sjf_aging_guarded`
- `sjf_aging_guarded_tail`
- `bypass_guard_sjf`
- `size_bucket_sjf_aging`
- `type_fifo_defer_budget`
- `slo_first`
- `p95-first`
- `p95-first-deadline`
- `p95-bucket-sjf`
- `p95-bucket-sjf-normalized`
- `p95-fusion`
- `slack_age`
- `slack_cost_age`
- `slack_hybrid`

CLI 入口：

```bash
--instance-scheduler-policy {fcfs,sjf,sjf_aging,sjf_aging_guarded,sjf_aging_guarded_tail,bypass_guard_sjf,size_bucket_sjf_aging,type_fifo_defer_budget,slo_first,p95-first,p95-first-deadline,p95-bucket-sjf,p95-bucket-sjf-normalized,p95-fusion,slack_age,slack_cost_age,slack_hybrid}
```

其中：

- `fcfs`
  - 严格按入队顺序调度
- `sjf`
  - 按估算剩余耗时从小到大排序
- `sjf_aging`
  - 在 `sjf` 的剩余耗时排序上叠加等待时长老化分数 `remaining_cost / (1 + aging_factor * cost_weight * age)`
  - 其中 `cost_weight = clip(sqrt(remaining_cost / 12.0), 1.0, 4.0)`，让大请求在相同等待时间下获得更快的老化补偿
  - 当 `instance_scheduler_aging_factor <= 0` 时，使用内建默认 aging 系数 `1.0`，保证不额外配参也具备防饥饿能力
  - 对 step chunk / chunk preemption 友好：chunk 重入队后继续按“剩余耗时 + 首次 arrival 老化”计算
- `sjf_aging_guarded`
  - 以 `sjf_aging` 的 cost-aware aging 为基础，但保护阈值不再固定；系统会用最近滑动窗口里已完成请求的 queue-wait 高分位学习 `learned_wait_guard_s`，再与 `2.0 * estimated_cost_s` 取更大值
  - protected 队列按到达时间优先，避免老的大请求继续被新来的短请求无限插队
  - protected 请求一旦开始执行，会直接跑完剩余 steps，用均值换取更低的 absolute p95
- `sjf_aging_guarded_tail`
  - 保留 `sjf_aging_guarded` 的 learned wait guard 与 protected 语义，但允许把极少数“非常老、非常重、且延后它能明显放行更多轻请求”的请求沉到队尾
  - 被沉降的请求数量以 `2%` 双层预算为目标，同时受全局 unique-request 预算和 arrival-window 预算共同约束；小样本阶段允许先借出 `1` 个 bootstrap sink slot
  - 当前版本重新保留了 `hard_escape` 回升机制，但默认阈值刻意设得很大，默认压测里通常不会触发；只有显式把阈值调低时，已沉降请求才会脱离 tail，并进入 `protected_hard_escape`
- `bypass_guard_sjf`
  - 默认仍按 cost-aware `sjf_aging` 排序，但每个请求会维护一个 `can_bypass ∈ {0,1}` 状态
  - 系统会从最近滑动窗口里学习 `learned_wait_guard_s`；当 `age_s >= max(learned_wait_guard_s, 2.0 * estimated_cost_s)` 时，请求会锁定为 `can_bypass=0`
  - `can_bypass=0` 的请求会按到达时间优先，并在首次拿到执行权后直接跑完，用非常直接的“禁止继续插队”语义保护尾部请求
- `size_bucket_sjf_aging`
  - 先按固定分辨率 bucket 分组，再在 bucket 内按 `remaining_cost / (1 + aging_factor * age)` 做 SJF + Aging 排序
  - aging 会按等待时间逐步提升大 bucket 请求，避免长期饥饿
- `type_fifo_defer_budget`
  - 先按 `(width, height, num_frames, total_steps, num_outputs)` 把请求分成少量 type，每个 type 内严格 FIFO
  - type 之间只比较各自队头；默认按队头 `aged_cost` 选下一条，利用“种类少”的结构而不打破类内先来先服务
  - `queue_wait_p95_s` 不再用固定 `45s~120s` 截断；当前实现直接学习原始 queue-wait p95，再用 `max(15s, 0.5 * current_queue_median_estimated_cost_s)` 作为动态 floor，避免冷启动时阈值过小
  - 对当前最重 type 的老队头，若 `age_s >= max(queue_wait_p95_s, 2.0 * estimated_cost_s)`、仍有更轻 type 在排队、且 `defer_relief_score > defer_harm_score`，可把该请求标记为 `deferred`
  - `deferred` 请求统一沉到队尾；当前实现里的 defer 预算是双层严格约束：
    - 全局 unique-request 预算：`len(deferred_unique_requests) <= floor(len(arrived_unique_requests) * instance_scheduler_type_fifo_defer_budget_ratio)`
    - arrival-based 滑动窗口预算：`len(window_deferred_unique_requests) <= floor(len(window_arrived_unique_requests) * ratio)`
    - 单次 waiting queue 重排时，实际 `defer_budget_limit` 会在 queue-local adaptive budget、global remaining budget、window remaining budget 三者中取最小值
  - 默认预算已经收紧到 `2%`；同时“过饿停止 defer”的阈值现在可配置，默认 multiplier 很大，压测里通常近似不触发，只有显式调低时才真正把最重 type 队头从 defer 候选里释放出来
  - 因为全局预算按 unique request 严格累计，所以整个 workload 结束后，被标记为 `deferred` 的请求总数不会超过 `ratio` 对应的总体比例
- `slo_first`
  - 先求可按时完成的 `on_time` 集合，再按 `slack / remaining_cost` 排序
- `p95-first-deadline`
  - 复用 `p95-first` 的 normalized 学习链路，但将 `target_latency_ms` 转成 synthetic deadline，再按 slack/deadline 压力排序
- `p95-bucket-sjf-normalized`
  - 保留 `p95-bucket-sjf` 的 bucket + SJF 结构，但把内部 `target_p95_ms` 替换成 `p95-first` 的 normalized target
- `p95-fusion`
  - 复用 `p95-first` 的 normalized 学习链路，但不做单一分数 greedy
  - 先把 heavy 请求抽成一个逻辑 `tail lane`，再在 `urgent -> short_normal -> tail_lane -> rest` 这几个集合之间做 bounded 轮转
  - 对 heavy 请求可附带 request 级 `chunk_budget_steps` override；极度 overdue 的 heavy 请求会被标记为本轮直接跑完剩余 steps
- `slack_age`
  - 直接对整个等待队列按 `slack - aging_factor * age` 做单队列排序
- `slack_cost_age`
  - 直接对整个等待队列按 `slack + 0.25 * remaining_cost - aging_factor * age` 做单队列排序
- `slack_hybrid`
  - 基于松弛度比值 `((deadline - now - swap_overhead) / T_rem)` 在两种模式之间切换
  - 若任一任务的 slack ratio 小于 `panic_threshold`，进入 Panic Mode，按 EDF 调度并优先紧急任务
  - 否则进入 Throughput Mode，按 `T_rem - aging_factor * wait_time` 做 SRPT + Aging 排序

### 1.0 `p95-first` 算法简介

`p95-first` 的目标不是最小化平均时延，而是优先压低实例内的尾时延。当前实现不再维护一个“全局绝对 p95 毫秒阈值”，而是把每个请求都映射到同一套 **runtime-normalized tail pressure** 框架里，再按压力大小重排等待队列。

可以把它理解成 4 个步骤：

1. 先把请求规模统一映射成 `work_units`
2. 再根据真实 chunk 执行时间在线学习当前实例速度
3. 然后给每个请求估算剩余纯执行时间 `estimated_service_ms`
4. 最后比较“它的预计完成时延”相对“它自己的 tail budget”有多危险

#### 1.0.1 输入量

当前实现里，一个请求的剩余 work 用 `work_units` 表示：

```text
work_units
= remaining_steps * num_frames * num_outputs
  * max(area / 1024^2, 0.0625)
```

这里的目标不是得到物理上精确的 FLOPs，而是构造一个在分辨率、步数、帧数、输出数之间单调一致的规模指标。

#### 1.0.2 在线学习量

`p95-first` 在线学习两个量：

1. `observed_service_rate_ms_per_work_unit`
   - 当前实例处理 1 个 `work_unit` 需要多少真实执行时间。
   - 每个已完成 chunk 都会产出一个 sample：

```text
observed_rate = chunk_execute_ms / chunk_work_units
```

   - 然后用 EMA 平滑更新：

```text
service_rate = (1 - alpha) * old_rate + alpha * observed_rate
```

2. `learned_slowdown_p95`
   - 已完成请求的 slowdown p95。
   - slowdown 定义为：

```text
slowdown = request_latency_ms / max(request_execute_ms, 1e-9)
```

   - 直观上，它表示“端到端 latency 相对纯执行时间被放大了多少”。

如果系统还处于冷启动阶段：

- `service_rate` 还没有真实 sample 时，回退到 `estimated_cost_s`
- `learned_slowdown_p95` 没有历史样本时，回退到 `1.0`

#### 1.0.3 单个请求的 tail pressure

对每个等待中的请求，先估算它的剩余纯执行时间：

```text
estimated_service_ms = service_rate * work_units
```

再估算“如果现在轮到它执行，它最终大概会在什么时候完成”：

```text
predicted_finish_latency_ms
= age_ms + cursor_ms + estimated_service_ms
```

其中：

- `age_ms` 是请求已经等待的时间
- `cursor_ms` 是它前面还要再等多久
- `estimated_service_ms` 是它自己的剩余纯执行时间

然后给这个请求一个属于它自己的 tail budget：

```text
target_latency_ms = learned_slowdown_p95 * estimated_service_ms
```

最后定义 tail pressure：

```text
pressure_ratio = predicted_finish_latency_ms / max(target_latency_ms, 1e-9)
risk_ms = predicted_finish_latency_ms - target_latency_ms
```

含义很直接：

- `pressure_ratio > 1` 表示这个请求预计会超过自己的 tail budget
- `pressure_ratio` 越大，说明它越值得被优先“救”

#### 1.0.4 排序分数与队列构造

当前实现的基础打分项是：

```text
base_score = -pressure_ratio
```

在此之上，还可以叠加 3 个修饰项：

- `size_bias`
- `age_bias`
- `starvation_boost`

最终分数为：

```text
final_priority_score
= -pressure_ratio
  + size_bias * (estimated_service_ms / 1000)
  - age_bias * age_s
  - starvation_boost
```

注意两点：

1. 这里不是一次性独立给每个请求打分后直接排序。
2. 当前实现会用 greedy 方式逐个挑选请求，并在每选中一个请求后，把它的 `estimated_service_ms` 累加进 `cursor_ms`，再重新评估剩余请求。

因此它优化的是“整条等待队列的归一化尾压顺序”，而不是单个请求的局部静态优先级。

#### 1.0.5 运行时边界

`p95-first` 只有在队列重排时才会重新生效：

- 新请求到达时
- unfinished 请求在 chunk 边界重新入队时

它不会打断 chunk 内部执行，所以当前模型仍然是：

- chunk 内连续执行
- chunk 边界重新评估 tail pressure
- 再决定下一轮 waiting queue 的顺序

这也是为什么它强依赖：

- `step chunk`
- `chunk preemption`

如果 chunk 太大：

- runtime sample 更新会变慢
- tail pressure 的响应会变钝
- 排序收益会下降

#### 1.0.6 一句话概括

当前仓库里的 `p95-first` 可以概括为：

> 先用真实运行数据学习实例当前速度和尾部 slowdown，再把每个请求映射成“预计完成时延 / 自身 tail budget”的压力值，最后按这个压力对等待队列做 greedy 重排。

### 1.1 `p95-first-deadline` 算法完整说明

`p95-first-deadline` 复用 `p95-first` 的 runtime-normalized 学习链路，但不直接做 tail-pressure greedy 排序。它先把每个请求的 normalized tail budget 转成一个内部 synthetic deadline，再按 deadline/slack 压力做单队列排序。

它学习的在线量与 `p95-first` 相同：

- `observed_service_rate_ms_per_work_unit`
  - 来自真实 chunk 执行时间的在线服务速率估计。
- `learned_slowdown_p95`
  - 来自完成请求 `latency / execute_time` 的 slowdown p95。
- `estimated_service_ms`
  - 当前请求剩余 work 在当前实例速度下的纯执行时间估计。

在此基础上，它先构造该请求自己的 normalized target：

```text
target_latency_ms = learned_slowdown_p95 * estimated_service_ms
```

然后把这个 target 直接转成调度器内部 deadline：

```text
synthetic_deadline_ts = arrival_time + target_latency_ms / 1000
```

这里的 deadline 不是请求协议字段，也不读取显式 `deadline_ts`。它是调度器根据当前实例实时速度和 slowdown 历史，为该请求内部派生出来的 deadline。

当前实现里还会计算：

- `availability_ts`
  - 当前实例最早可开始处理等待队列的时间点。
  - 与 deadline-aware 策略保持一致，当前实现使用 active request 的 total remaining cost 推到该时间点。
- `slack_s`
  - 如果该请求作为下一批等待队列中的候选被安排，离 synthetic deadline 还剩多少余量。

```text
slack_s = synthetic_deadline_ts - availability_ts - estimated_service_ms / 1000
```

- `urgency_ms`
  - 请求的 synthetic deadline 相对当前 availability 的窗口宽度。

```text
urgency_ms = (synthetic_deadline_ts - availability_ts) * 1000
```

最终排序键为：

```text
sort_key = (
    slack_s,
    synthetic_deadline_ts,
    estimated_service_ms,
    enqueue_time,
    sequence_id,
)
```

这意味着：

- 先处理 synthetic slack 更小的请求。
- slack 接近时，先处理 synthetic deadline 更早的请求。
- 仍然保留 `estimated_service_ms` 作为 tie-break，让更短的任务更容易在同类压力下先过。

直观理解：

- `p95-first` 是“谁更接近撞上自己的 normalized tail budget，就更值得立刻救”。
- `p95-first-deadline` 是“先把这个 normalized tail budget 固化成内部 ddl，再按 ddl/slack 语义去排队”。

所以它适合这类场景：

- 你认可 `p95-first` 的 runtime-normalized 学习方法。
- 但你希望调度器输出的是更容易解释的 deadline/slack 语义，而不是单一 pressure score。
- 你后续还想把同一套 normalized ddl 估计迁移给别的 deadline-aware 策略。

与运行时机制的关系：

- 该策略同样依赖 step chunk + chunk preemption 才能在 chunk 边界持续重排。
- active request 不会在 chunk 内被打断。
- 如果实例还没有 runtime sample，会回退到 `estimated_cost_s * 1000` 作为 `estimated_service_ms`。

### 1.2 `p95-bucket-sjf` 算法完整说明

`p95-bucket-sjf` 不是显式 SLO 调度器，也不是单一分数调度器。它假设请求侧最少只提供 `estimated_cost_s`，然后在实例内本地派生一个“synthetic deadline”，再把紧迫度接近的请求放入同一个 bucket，最后在 bucket 内执行 SJF。

它的目标不是绝对公平，也不是简单最小化平均时延，而是在“尾时延保护”和“短作业吞吐”之间做结构化折中：

1. 先用历史 p95 给每个请求构造一个目标完成时间。
2. 再用这个目标时间得到 deadline-aware 的粗粒度优先级。
3. 最后在同一优先层内继续偏向短作业。

当前实现里的关键量如下：

- `estimated_cost_s`
  - 请求的剩余成本。
  - 若请求显式提供 `sampling_params.extra_args["estimated_cost_s"]`，调度器直接使用，并按剩余 steps 比例缩放。
  - 若请求未提供，则回退到 runtime profile 或启发式估算。
- `history_p95_ms`
  - 实例内近期完成请求的 learned p95。
  - 当前实现直接复用 `p95-first` 的历史队列和 cold-start fallback，不单独维护第二套历史。
- `target_p95_ms`
  - 针对单个请求生成的本地 tail target。

```text
target_p95_ms = max(history_p95_ms, estimated_cost_s * 1000)
```

这样可以保证：

- 小请求不会完全脱离实例当前 tail 压力。
- 大请求不会因为历史 p95 偏小而拿到明显不合理的过短目标。

- `deadline_ts`
  - 不是请求协议字段，而是调度器内部派生值。

```text
deadline_ts = arrival_ts + target_p95_ms / 1000
```

- `availability_ts`
  - 当前实例最早能开始处理等待队列的时间点。
  - 若存在 active request，当前实现使用 active request 的 total remaining cost，而不是仅仅看当前 chunk blocking。

```text
availability_ts = now + active_total_remaining_cost_s
```

- `urgency_ms`
  - 表示该请求距离 synthetic deadline 还剩多少可用时间。

```text
urgency_ms = (deadline_ts - availability_ts) * 1000
```

其中：

- `urgency_ms` 越小表示越紧急。
- `urgency_ms <= 0` 表示按当前估计已经进入“应尽快处理”的区间。

- `bucket_width_ms`
  - 当前批次等待队列中每个 bucket 覆盖的紧迫度区间宽度。

```text
anchor_window_ms = max(history_p95_ms, max_estimated_cost_ms_in_queue, min_bucket_window_ms)
bucket_width_ms = anchor_window_ms / bucket_count
```

- `raw_bucket_id`
  - 根据紧迫度落桶后的原始 bucket。

```text
raw_bucket_id =
    0,                                   if urgency_ms <= 0
    min(floor(urgency_ms / bucket_width_ms), bucket_count - 1), otherwise
```

- `effective_bucket_id`
  - 应用饥饿保护之后的 bucket。

```text
effective_bucket_id = max(raw_bucket_id - promote_levels, 0)
```

只有当：

```text
age_s >= instance_scheduler_p95_bucket_starvation_threshold_s
```

时，才会发生 bucket promotion。

最终排序键不是一个混合分数，而是一个两级结构：

```text
sort_key = (
    effective_bucket_id,
    estimated_cost_s,
    enqueue_time,
    sequence_id,
)
```

这意味着：

- 先保证更紧急 bucket 的请求整体前移。
- 再保证同一 bucket 内短作业优先。
- 最后用入队顺序和序号做稳定 tie-break。

直观理解：

- `p95-first` 是“全队列比较谁更值得救”。
- `p95-bucket-sjf` 是“先把请求按 deadline 压力分层，再在层内做 SJF”。

所以它比 `p95-first` 更容易解释，也更接近“少量优先级层 + 每层 SRPT/SJF”的工程实现。

与运行时机制的关系：

- 该策略同样依赖 step chunk + chunk preemption 才能持续重排。
- active request 不会在 chunk 内被打断，只会在 chunk 边界重新入队后参与下一轮 bucket 计算。
- 因此它是“chunk 边界上的 synthetic-deadline bucketed scheduling”，不是连续抢占模型。

适用边界：

- 混合长短任务且只愿意从请求侧提供 `estimated_cost_s` 的场景。
- 希望比 `p95-first` 更稳定、更容易排障的 tail-aware 调度场景。
- 如果 runtime profile 很差，或 `estimated_cost_s` 明显失真，bucket 划分质量会直接下降。
- 如果系统长期重载，很多请求会一起掉进最紧急 bucket，此时策略会退化得更像“加了饥饿保护的 SJF”。

### 1.3 `p95-bucket-sjf-normalized` 算法完整说明

`p95-bucket-sjf-normalized` 是 `p95-bucket-sjf` 的结构化副本：bucket 划分、bucket 内 SJF、starvation promotion 都保持不变，唯一变化是内部的 p95 估计不再来自绝对 latency history，而是复用 `p95-first` 的 runtime-normalized 学习链路。

它对每个请求先计算：

```text
estimated_service_ms = service_rate * work_units
target_latency_ms = learned_slowdown_p95 * estimated_service_ms
synthetic_deadline_ts = arrival_time + target_latency_ms / 1000
urgency_ms = (synthetic_deadline_ts - availability_ts) * 1000
```

也就是说：

- 旧 `p95-bucket-sjf` 用的是 absolute `history_p95_ms`。
- 新 `p95-bucket-sjf-normalized` 用的是 `learned_slowdown_p95 * estimated_service_ms`。

之后它仍然沿用 bucket 化逻辑：

```text
bucket_width_ms = max(max_target_latency_ms_in_queue, min_bucket_window_ms) / bucket_count
```

```text
raw_bucket_id =
    0, if urgency_ms <= 0
    min(floor(urgency_ms / bucket_width_ms), bucket_count - 1), otherwise
```

最后排序键仍是两级结构：

```text
sort_key = (
    effective_bucket_id,
    estimated_service_ms,
    enqueue_time,
    sequence_id,
)
```

所以它的工程语义很直接：

- deadline 压力来自 `p95-first` 的 normalized p95 估计。
- 可解释性仍然保持在 `p95-bucket-sjf` 这一层，而不是回到连续分数排序。
- 如果你想保留 bucket 化 tail protection，但又不想继续依赖 absolute p95 history，它就是对应副本。

### 1.4 `sjf_aging_guarded_tail` 算法完整说明

`sjf_aging_guarded_tail` 可以理解为 `sjf_aging_guarded` 的有界让渡版本：大多数老请求仍然会被放进 protected 队列优先保障，但对于极少数“已经很老、又明显比队列里其他请求更重”的请求，调度器允许把它暂时沉到队尾，并把它当成一个可被空闲时间消费的 background tail job。

这个策略的目标不是否定 guarded 保护，而是引入一个工程上更保守的 tradeoff：

- 默认仍然保护老请求
- 只有极少数 super-heavy 请求会被让渡
- 让渡比例必须有硬预算，不允许无限扩大
- 让渡过头时要能自动逃逸回 protected 语义

#### 1.4.1 第一阶段仍然是 guarded 保护

当前实现第一步与 `sjf_aging_guarded` 相同，先计算：

```text
aged_cost_s
learned_wait_guard_s
protection_threshold_s = max(learned_wait_guard_s, 2.0 * estimated_cost_s)
tail_protected = age_s >= protection_threshold_s
```

也就是说，request 只有先成为 `tail_protected`，后面才有资格进入“是否沉降”的第二阶段判定。

#### 1.4.2 第二阶段只筛 very old super-heavy 请求

在 protected 请求里，当前实现再额外判断两件事：

1. 它是不是当前队列里的 `super_heavy`
2. 它是不是已经老到允许暂时让渡

当前实现里的重请求判定来自 queue-local 成本分布：

```text
large_request_threshold_s = max(1.5 * queue_median_cost_s, queue_p75_cost_s)
super_heavy = estimated_cost_s >= large_request_threshold_s
```

沉降阈值不再要求请求在进入 protected 之后再额外多等一大截，而是直接贴近 protected 边界：

```text
sink_threshold_s = max(protection_threshold_s, 1.5 * estimated_cost_s)
hard_escape_threshold_s = max(100.0 * learned_wait_guard_s, 100.0 * estimated_cost_s)
```

当前版本里，`hard_escape` 已重新接回真实调度路径，但默认阈值被故意抬得非常高，目的是在默认 benchmark 配置下近似“不起作用”，只有在实验里显式把 multiplier 调低时才真正参与回升。

#### 1.4.3 候选沉降不再用 `relief > harm` 做硬门槛

当前实现仍然会计算：

```text
defer_relief_score = lighter_request_count * max(estimated_cost_s - lighter_mean_cost_s, 0)
defer_harm_score = estimated_cost_s * max(age_s / hard_escape_threshold_s, 1.0)
```

但它们的用途已经调整成：

- `defer_relief_score`
  - 主要用于在多个可沉降请求之间做排序，优先挑“沉下去能放行更多 lighter requests”的候选
- `defer_harm_score`
  - 主要用于观测和解释，不再作为必须满足的硬门槛

真实 gate 现在更偏工程规则：

- `tail_protected`
- `super_heavy`
- `age_s >= sink_threshold_s`
- `lighter_request_count >= 1`
- `len(waiting_queue) >= 3`

#### 1.4.4 2% 预算与滑动窗口

这是该策略和 `sjf_aging_guarded` 最大的行为差异。当前实现对被沉降的 unique requests 施加了双重硬预算：

- 全局 unique-request 预算
- arrival-window unique-request 预算

默认形式分别是：

```text
global_used <= floor(global_arrived_unique * 0.02)
window_used <= floor(window_arrived_unique * 0.02)
```

其中 arrival window 使用一个固定长度窗口追踪最近到达的请求，避免只看当前 waiting queue 太小而完全失去比例感知。

为了避免小样本阶段永远得不到任何沉降机会，当前实现还额外加了一个 bootstrap 规则：

```text
if arrived_unique >= 2:
    budget_limit = max(floor(arrived_unique * 0.02), 1)
```

也就是说：

- 当全局或窗口里还没有累计到足够多 unique requests 时，允许先借出 1 个 bootstrap sink slot
- 一旦这个 slot 用掉，后续仍然继续受全局 + 窗口预算约束，不会无限放大

当前每轮 waiting queue 重排还会额外收紧成：

```text
tail_defer_budget_limit = min(1, global_budget_remaining, window_budget_remaining)
```

也就是说：

- 单轮最多只沉 1 个请求
- 大样本下整体被沉降的 unique request 数量不会超过 2%
- 小样本下最多只会先借出 1 个 bootstrap slot

#### 1.4.5 最终队列顺序

最终排序现在保持四层语义：

1. `protected_hard_escape`
2. `normal`
3. `protected_soft`
4. `sunk_tail`

其中：

- normal 请求继续按 `aged_cost_s` 排序
- `soft protected` 不再绝对压过 normal，而是退到 normal 之后，内部继续按 arrival/FIFO
- sunk 请求统一沉到底部，内部保持 arrival/FIFO 顺序
- 已经 sunk 的请求会在后续 reorder 中保持 sunk；即使它被 dequeue 执行了一个 chunk，只要后续又 requeue 回 waiting queue，仍会继续以 sunk 身份回到队尾
- 当前版本里，sunk 状态默认只会在 `finish`、`fail`、`abort` 等请求终止语义下解除；只有显式调低 `hard_escape` 阈值并真正命中时，sunk 请求才会脱离 tail
- `sjf_aging_guarded_tail` 默认仍按 chunk 参与调度；只有命中 `hard_escape` 的非 sunk protected 请求才会恢复 run-to-completion

#### 1.4.6 一句话概括

当前仓库里的 `sjf_aging_guarded_tail` 可以概括为：

> 先沿用 `sjf_aging_guarded` 保护老请求，再把极少数 very old super-heavy 请求在双层预算约束下沉到底部，并把它们当成 sticky tail background jobs：队列空闲时照常跑，有新请求时再回到最后；同时保留一个默认阈值极高的 `hard_escape` 兜底，只有实验里显式调低阈值时才会真正回升。

### 1.5 `p95-fusion` 算法完整说明

`p95-fusion` 可以理解为“在 `p95-first` 的 normalized tail estimation 之上，再额外加一层工程化的 heavy-request 保护”。它不引入第二套绝对 p95 历史，而是继续复用：

- `estimated_service_ms`
- `learned_slowdown_p95`
- `dynamic_p95_ms`
- `active_chunk_blocking_ms`

在此基础上，它做 4 件额外的事：

1. 识别 heavy 请求，构造一个逻辑 `tail lane`
2. 识别已经接近或超过自身 tail budget 的 urgent 请求
3. 在短请求和 heavy 请求之间做有限轮转，避免 heavy 长期饥饿
4. 给 heavy 请求下发 request 级 chunk override，必要时直接切到 run-to-completion

#### 1.5.1 请求分类

对每个 waiting request，当前实现先复用 normalized 链路得到：

```text
estimated_service_ms
target_latency_ms
predicted_finish_latency_ms
slack_s
slack_ratio
pressure_ratio
```

然后做两层分类：

```text
is_heavy  = estimated_service_s >= heavy_threshold_s
is_urgent = (slack_s <= 0)
         or (slack_ratio <= urgent_slack_ratio)
         or (is_heavy and age_s >= promote_wait_s)
```

含义分别是：

- `is_heavy`
  - 剩余纯执行时间已经超过 heavy 阈值，应该纳入尾部保护候选
- `is_urgent`
  - 已经明显撞线，或虽然还没撞线但 slack 非常紧，或者 heavy 请求已经等太久

#### 1.5.2 逻辑 `tail lane`

所有 heavy 请求都会先进入 `tail_candidate` 集合，然后按以下顺序排序：

1. urgent heavy 优先
2. `slack_s` 更小优先
3. `age_s` 更大优先
4. `enqueue_time` 更早优先

随后根据 lane 容量截断成真正的 `tail_lane`：

```text
ratio_cap = ceil(queue_len * tail_budget_ratio)
borrowed_cap = min(current_borrowed_cap, borrowed_cap_max)
lane_cap = min(max(ratio_cap, borrowed_cap), ceil(queue_len * 0.35))
```

这里的 `borrowed_cap` 会随着 arrival 数增长而逐步放大，使 backlog 变大时能允许更多 heavy 请求进入保护通道；同时又会被 `0.35 * queue_len` 的硬上限约束住，不让 tail lane 吞掉整个队列。

#### 1.5.3 最终队列构造

当前实现不会把队列简化成一个静态打分排序，而是先拆成 4 个集合：

- `urgent_all`
  - 所有 urgent 请求，按 `slack_s`、`estimated_service_ms`、到达顺序排序
- `short_normal`
  - 非 urgent 且非 heavy 的请求，按 `estimated_service_ms` 优先
- `nonurgent_tail_lane`
  - 已进入 tail lane 且当前不是 urgent 的 heavy 请求
- `rest`
  - 其他未进入以上集合的请求，按 `pressure_ratio` 倒序补位

然后按下面的 bounded 轮转逻辑出队：

1. 如果有 `urgent_all`，总是先取 urgent
2. 否则若 `nonheavy_streak >= nonheavy_streak_limit` 且 `pending_tail` 非空，强制给一次 heavy turn
3. 否则优先取 `short_normal`
4. 再取 `pending_tail`
5. 最后取 `pending_rest`

这意味着它的目标不是“只救 heavy”，而是：

- 紧急请求永远优先
- 短请求在正常情况下保持较高吞吐
- heavy 请求不会因为队列中持续有 short job 而被无限推迟

#### 1.5.4 request 级 chunk override

`p95-fusion` 的一个关键区别是，它不只重排 waiting queue，还会直接给 heavy 请求打 request 级执行提示：

- 如果 heavy 请求已经极度 overdue，满足

```text
is_urgent and overdue_s >= estimated_service_s
```

  - 则设置 `scheduler_force_run_to_completion=True`
- 否则如果它是 heavy 且仍有剩余 steps
  - 则设置 `scheduler_chunk_budget_steps`
  - 当前范围由
    - `instance_scheduler_p95_fusion_min_chunk_steps`
    - `instance_scheduler_p95_fusion_max_chunk_steps`
    共同约束

这些 override 会直接回写到 request 对象，再由 `DiffusionEngine` 在本轮 chunk 规划时读取。

#### 1.5.5 一句话概括

当前仓库里的 `p95-fusion` 可以概括为：

> 先沿用 `p95-first` 的 normalized tail estimation 识别 heavy / urgent 请求，再用逻辑 tail lane 和 bounded heavy/non-heavy 轮转保护大请求尾部，同时通过 request 级 chunk override 让 heavy 请求以更细粒度或直接跑完的方式落地。

### 1.6 各策略可调参数与默认值

| 策略 | 主要排序逻辑 | 可调参数 | 默认值 |
| --- | --- | --- | --- |
| `fcfs` | 按 `enqueue_time` 先来先服务 | 无 | 无 |
| `sjf` | 按 `estimated_cost_s` 从小到大排序 | 无 | 无 |
| `sjf_aging` | 按 `estimated_cost_s / (1 + aging_factor * cost_weight * age_s)` 排序，其中 `cost_weight = clip(sqrt(estimated_cost_s / 12.0), 1.0, 4.0)` | `instance_scheduler_aging_factor` | `0.0`<br>实现中当 `<= 0` 时实际回退为内建 aging 因子 `1.0` |
| `sjf_aging_guarded` | 在 cost-aware `sjf_aging` 上增加 protected 队列；当 `age_s >= max(learned_wait_guard_s, 2.0 * estimated_cost_s)` 时转入 protected，并按 `arrival_time` 优先；protected 请求开始执行后直接跑完 | `instance_scheduler_aging_factor` | `0.0`<br>`learned_wait_guard_s` 由最近滑动窗口自动学习，当前代码内 floor=45s、cap=120s |
| `sjf_aging_guarded_tail` | 保留 `sjf_aging_guarded` 的 protected 队列，但允许把极少数 very old super-heavy 请求按 bounded rule 沉到 tail；未 sunk 的 protected 统一退到 normal 之后；已经 sunk 的请求会跨 chunk 保持 sticky tail 语义；一旦命中 hard escape，会脱离 tail 并切回最高优先级 | `instance_scheduler_aging_factor`<br>`instance_scheduler_sjf_aging_guarded_tail_defer_budget_ratio`<br>`instance_scheduler_sjf_aging_guarded_tail_hard_escape_wait_multiplier`<br>`instance_scheduler_sjf_aging_guarded_tail_hard_escape_cost_multiplier` | `0.0`<br>`0.02`<br>`100.0`<br>`100.0`<br>`learned_wait_guard_s` 由最近滑动窗口自动学习；沉降预算目标为 `2%`，但在全局/窗口 unique arrivals 至少为 `2` 时允许先借出 `1` 个 bootstrap sink slot；queue-local super-heavy 阈值为 `max(1.5 * queue_median_cost_s, queue_p75_cost_s)`；sink threshold 为 `max(protection_threshold_s, 1.5 * estimated_cost_s)`；默认 `hard_escape` 阈值很高，压测中通常近似关闭；当请求命中 `hard_escape` 时会脱离 tail，并在执行层恢复 run-to-completion |
| `bypass_guard_sjf` | 默认按 cost-aware `sjf_aging` 排序；当 `age_s >= max(learned_wait_guard_s, 2.0 * estimated_cost_s)` 时将 `can_bypass` 置为 `0`，锁定请求不再允许被后到请求插队，并在 dispatch 后直接跑完 | `instance_scheduler_aging_factor` | `0.0`<br>`learned_wait_guard_s` 由最近滑动窗口自动学习，当前代码内 floor=45s、cap=120s |
| `size_bucket_sjf_aging` | 先按固定分辨率 bucket 排序，再在 bucket 内按 `estimated_cost_s / (1 + aging_factor * age_s)` 排序；等待过久时允许跨 bucket 晋升 | `instance_scheduler_aging_factor` | `0.0`<br>实现中当 `<= 0` 时实际回退为内建 aging 因子 `1.0` |
| `type_fifo_defer_budget` | 先按 request type 分 FIFO 子队列；type 间只比较队头 `aged_cost_s`；对最重 type 的老队头做 bounded defer，但只有在 `defer_relief_score > defer_harm_score` 且未进入 hard-escape/over-starved 区间时才允许延后 | `instance_scheduler_aging_factor`<br>`instance_scheduler_type_fifo_defer_budget_ratio`<br>`instance_scheduler_type_fifo_defer_hard_escape_wait_multiplier`<br>`instance_scheduler_type_fifo_defer_hard_escape_cost_multiplier` | `0.0`<br>`0.02`<br>`100.0`<br>`100.0`<br>`queue_wait_p95_s` 直接学习原始 queue-wait p95，不再固定 cap=120s；当前只使用动态 floor `max(15s, 0.5 * current_queue_median_estimated_cost_s)`；defer threshold 为 `max(queue_wait_p95_s, 2.0 * estimated_cost_s)`；hard-escape/over-starved threshold 为 `max(wait_multiplier * queue_wait_p95_s, cost_multiplier * estimated_cost_s)`；默认 multiplier 很大，压测中通常近似关闭；`ratio` 同时充当全局 unique-request 牺牲上限和 arrival-based 滑动窗口上限，单次重排只会在这两层剩余预算内再取 queue-local budget |
| `slo_first` | 先求 `on_time` 集合，再按 `slack / remaining_cost` 排序；尾部按 aging best-effort 排序 | `instance_scheduler_slo_target_ms`<br>`instance_scheduler_slo_floor_ms`<br>`instance_scheduler_aging_factor` | `None`<br>`0.0`<br>`0.0` |
| `p95-first` | 基于 `learned_slowdown_p95`、`estimated_service_ms`、`pressure_ratio` 的 normalized tail-pressure 单队列排序 | `instance_scheduler_p95_first_size_bias`<br>`instance_scheduler_p95_first_age_bias`<br>`instance_scheduler_p95_first_starvation_threshold_s`<br>`instance_scheduler_p95_first_starvation_boost`<br>`instance_scheduler_p95_first_base_ms`（兼容保留）<br>`instance_scheduler_p95_first_min_ms`（兼容保留）<br>`instance_scheduler_p95_first_max_ms`（兼容保留）<br>`instance_scheduler_p95_first_backlog_alpha`（兼容保留） | `0.0`<br>`0.0`<br>`None`<br>`0.0`<br>`None`<br>`0.0`<br>`None`<br>`1.0` |
| `p95-first-deadline` | 复用 `learned_slowdown_p95` 与 `estimated_service_ms` 生成 `synthetic_deadline_ts`，再按 `slack_s` / `deadline` / `estimated_service_ms` 排序 | `instance_scheduler_p95_first_base_ms`（兼容保留）<br>`instance_scheduler_p95_first_min_ms`（兼容保留）<br>`instance_scheduler_p95_first_max_ms`（兼容保留）<br>`instance_scheduler_p95_first_backlog_alpha`（兼容保留） | `None`<br>`0.0`<br>`None`<br>`1.0` |
| `p95-bucket-sjf` | 先按 `urgency_ms` 划 bucket，再在 bucket 内按 `estimated_cost_s` 做 SJF | `instance_scheduler_p95_bucket_count`<br>`instance_scheduler_p95_bucket_min_window_ms`<br>`instance_scheduler_p95_bucket_starvation_threshold_s`<br>`instance_scheduler_p95_bucket_starvation_promote_levels`<br>`instance_scheduler_p95_first_min_ms` | `4`<br>`200.0`<br>`None`<br>`1`<br>`0.0` |
| `p95-bucket-sjf-normalized` | 先按 normalized `urgency_ms` 划 bucket，再在 bucket 内按 `estimated_service_ms` 做 SJF | `instance_scheduler_p95_bucket_count`<br>`instance_scheduler_p95_bucket_min_window_ms`<br>`instance_scheduler_p95_bucket_starvation_threshold_s`<br>`instance_scheduler_p95_bucket_starvation_promote_levels` | `4`<br>`200.0`<br>`None`<br>`1` |
| `p95-fusion` | 复用 normalized p95 估计识别 heavy/urgent，请求先拆成 `urgent`、`short_normal`、`tail_lane`、`rest`，再做 bounded heavy/non-heavy 轮转；同时可对 heavy 请求下发 request 级 chunk override | `instance_scheduler_p95_fusion_tail_budget_ratio`<br>`instance_scheduler_p95_fusion_heavy_threshold_s`<br>`instance_scheduler_p95_fusion_urgent_slack_ratio`<br>`instance_scheduler_p95_fusion_promote_wait_s`<br>`instance_scheduler_p95_fusion_nonheavy_streak_limit`<br>`instance_scheduler_p95_fusion_growth_every`<br>`instance_scheduler_p95_fusion_borrowed_cap_max`<br>`instance_scheduler_p95_fusion_min_chunk_steps`<br>`instance_scheduler_p95_fusion_max_chunk_steps` | `0.10`<br>`20.0`<br>`1.0`<br>`60.0`<br>`4`<br>`20`<br>`4`<br>`1`<br>`8` |
| `slack_age` | 单队列按 `slack - aging_factor * age_s` 排序 | `instance_scheduler_aging_factor` | `0.0` |
| `slack_cost_age` | 单队列按 `slack + 0.25 * remaining_cost_s - aging_factor * age_s` 排序 | `instance_scheduler_aging_factor` | `0.0`<br>其中 `0.25` 是当前代码内常量，不是配置项 |
| `slack_hybrid` | 若任一请求 `slack_ratio < panic_threshold`，切到 Panic EDF；否则按 `estimated_cost_s - aging_factor * age_s` 做 Throughput SRPT + Aging 排序 | `instance_scheduler_aging_factor`<br>`instance_scheduler_slack_panic_threshold`<br>`instance_scheduler_slack_swap_overhead_ms` | `0.0`<br>`1.0`<br>`0.0` |

补充：

- `estimated_cost_s` 不是实例级配置，而是请求运行时输入；若请求未提供，调度器会回退到 runtime profile 或启发式估算。
- `deadline_ts`、`slo_target_ms`、`slo_ms` 也是请求运行时输入；若 deadline-aware 策略没有显式 deadline，会回退到 learned-p95 synthetic deadline。
- `p95-first`、`p95-first-deadline`、`p95-bucket-sjf`、`p95-bucket-sjf-normalized`、`p95-fusion`、`slack_hybrid`、`sjf_aging_guarded`、`sjf_aging_guarded_tail`、`bypass_guard_sjf`、`type_fifo_defer_budget` 在当前实现里会自动启用：
  - `diffusion_enable_step_chunk=True`
  - `diffusion_enable_chunk_preemption=True`

## 2. 调度器的行为边界

当前 `Stage1Scheduler` 只有一个 active request。它不会抢占已经在执行中的 chunk，只会在以下时机重排等待队列：

- 新请求入队
- 未完成请求在 chunk 边界重新入队

这意味着：

- 如果未启用 step chunk / chunk preemption，等待队列仍然可以重排
- 但已经开始执行的请求会一直跑到本次 worker 返回，期间不会被中断
- 想让 deadline-aware 策略持续生效，通常需要配合 chunk 级重入队

相关配置：

```bash
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption
```

约束关系：

- `diffusion_enable_chunk_preemption=True` 要求 `diffusion_enable_step_chunk=True`

补充说明：

- `diffusion_small_request_latency_threshold_ms` 命中时，请求会直接跑完剩余 steps，不再切 chunk
- `diffusion_chunk_budget_steps`
- `diffusion_image_chunk_budget_steps`
- `diffusion_video_chunk_budget_steps`
  - 这些参数决定每次重新入队前最多执行多少 steps

## 3. 启用不同策略时需要哪些参数

### 3.1 `fcfs`

不需要额外参数。

### 3.2 `sjf` / `sjf_aging` / `sjf_aging_guarded` / `sjf_aging_guarded_tail` / `bypass_guard_sjf` / `size_bucket_sjf_aging` / `type_fifo_defer_budget`

这几种策略都不要求提供 deadline，但强烈建议提供更准确的耗时估计来源。当前耗时估计优先级是：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

其中 `sjf_aging_guarded` 会维护一个滑动窗口学习得到的 `learned_wait_guard_s`，用于决定请求何时进入不可继续插队的 protected 状态；`sjf_aging_guarded_tail` 先复用完全相同的 protected 判定，再对其中 very old super-heavy 的请求引入一个 bounded tail sink：只要它已经进入 protected、足够重、超过 sink threshold、队列里确实存在 lighter requests，就允许把它暂时沉到队尾，而且单轮最多只沉 1 个 unique request，并同时受全局 unique-request 预算与 arrival-window 预算这两层约束；为了避免小样本下预算始终为 0，这条策略在全局/窗口 unique arrivals 至少为 `2` 时允许先借出 `1` 个 bootstrap sink slot。已经 sunk 的请求会保持 sticky tail 语义：队列空闲时它可以被 dispatch 跑一个 chunk，但只要它后续 requeue 回 waiting queue，仍会继续回到尾部，直到请求 finish/fail/abort 为止；当前版本重新保留了 `hard_escape` 回升与 RTC 兜底，但默认 multiplier 很大，压测里通常近似关闭；`bypass_guard_sjf` 也会学习同样语义的等待阈值，但它直接维护 `can_bypass ∈ {0,1}`，一旦降为 `0` 就不再允许新请求继续插队；`type_fifo_defer_budget` 会先按 `(width, height, num_frames, total_steps, num_outputs)` 聚成少量 type，并要求同 type 内严格 FIFO，只允许把最重 type 的极少数老队头延后到 bounded tail；defer 阈值来自该策略自身完成请求的 queue-wait 滑动窗口 p95 学习，形式为 `max(queue_wait_p95_s, 2.0 * estimated_cost_s)`。当前 `queue_wait_p95_s` 不再使用固定 `45s/120s` 截断，只保留一个动态 floor：`max(15s, 0.5 * current_queue_median_estimated_cost_s)`，这样高 backlog 下真实 tail 可以继续上升，不会被内部常量硬截断；同时它还会计算 `defer_relief_score` 与 `defer_harm_score`，只有“延后它能救下更多 lighter 请求”时才 defer；而它的 hard-escape/over-starved 阈值现在也改成可配置，默认 multiplier 很大，压测里通常近似关闭。预算上，这条策略同时维护全局 unique-request 预算和 arrival-based 滑动窗口预算，在大样本阶段会收敛到 `ratio` 对应的总体比例；而 `size_bucket_sjf_aging` 额外会先按固定分辨率 bucket 分组：

- `max(width, height) <= 512`
- `512 < max(width, height) <= 768`
- `768 < max(width, height) <= 1024`
- `max(width, height) > 1024`

然后在 bucket 内做 `SJF + Aging`，并按等待时长逐步把大 bucket 请求往前提升。

### 3.3 `p95-first` / `p95-first-deadline` / `p95-bucket-sjf` / `p95-bucket-sjf-normalized` / `p95-fusion`

这四种策略都只要求请求侧尽量提供更准确的 `estimated_cost_s`，不要求请求显式提供 `slo_ms`、`slo_target_ms` 或 `deadline_ts`。

其中：

- `p95-first`
  - 当前使用 normalized tail-pressure 单队列排序。
  - 它会从真实 chunk 运行时间学习实例服务速率，再从完成请求的 `latency / execute_time` 学习 `learned_slowdown_p95`。
  - 每个请求的 tail budget 由 `target_latency_ms = learned_slowdown_p95 * estimated_service_ms` 派生，不再依赖固定绝对毫秒目标。
  - 因此它更适合“不希望按数据集去调绝对 p95 毫秒参数”的场景。
  - 推荐同时配置：
    - `--instance-runtime-profile-path`
    - `--instance-runtime-profile-name`
    - `--diffusion-enable-step-chunk`
    - `--diffusion-enable-chunk-preemption`
  - `--instance-scheduler-p95-first-base-ms` / `min-ms` / `max-ms` / `backlog-alpha` 仍保留在配置层，但当前不参与 `p95-first` 排序决策。
- `p95-first-deadline`
  - 复用 `p95-first` 的 runtime-normalized 学习链路，但把 `target_latency_ms` 转成 `synthetic_deadline_ts`。
  - 当前排序直接比较 `slack_s`、`synthetic_deadline_ts` 和 `estimated_service_ms`，因此语义更接近 deadline-aware 调度。
  - 它不读取请求显式 `deadline_ts`，内部 ddl 完全由 normalized p95 学习结果派生。
  - 推荐同时配置：
    - `--instance-runtime-profile-path`
    - `--instance-runtime-profile-name`
    - `--diffusion-enable-step-chunk`
    - `--diffusion-enable-chunk-preemption`
- `p95-bucket-sjf`
  - 本地学习历史 `history_p95_ms`。
  - 对每个请求计算 `target_p95_ms = max(history_p95_ms, estimated_cost_ms)`。
  - 用 `deadline_ts = arrival_ts + target_p95_ms / 1000` 派生内部 deadline。
  - 按 `urgency_ms = (deadline_ts - availability_ts) * 1000` 做 bucket 划分。
  - bucket 内按 `estimated_cost_s` 做 SJF 排序。
  - 推荐同时配置：
    - `--instance-runtime-profile-path`
    - `--instance-runtime-profile-name`
    - `--diffusion-enable-step-chunk`
    - `--diffusion-enable-chunk-preemption`
- `p95-bucket-sjf-normalized`
  - 保留与 `p95-bucket-sjf` 相同的 bucket + starvation promotion + bucket 内 SJF 结构。
  - 但内部 target 不再来自 `history_p95_ms`，而是来自 `target_latency_ms = learned_slowdown_p95 * estimated_service_ms`。
  - 用 `synthetic_deadline_ts = arrival_ts + target_latency_ms / 1000` 派生 normalized deadline。
  - 按 normalized `urgency_ms` 做 bucket 划分，bucket 内按 `estimated_service_ms` 排序。
  - 推荐同时配置：
    - `--instance-runtime-profile-path`
    - `--instance-runtime-profile-name`
    - `--diffusion-enable-step-chunk`
    - `--diffusion-enable-chunk-preemption`
- `p95-fusion`
  - 复用 `p95-first` 的 normalized 学习链路，不要求请求显式提供 deadline。
  - 当前实现把请求拆成 `urgent`、`short_normal`、`tail_lane`、`rest` 4 个集合，再在这些集合之间做 bounded 轮转。
  - heavy 判定使用 `estimated_service_s >= instance_scheduler_p95_fusion_heavy_threshold_s`。
  - 若 heavy 请求满足 `slack_ratio` 过低、已经 overdue，或等待时间过长，会被标记为 urgent。
  - 对 heavy 请求，调度器还会下发 request 级 `scheduler_chunk_budget_steps`；极度 overdue 时会切到 `scheduler_force_run_to_completion=True`。
  - 推荐同时配置：
    - `--instance-runtime-profile-path`
    - `--instance-runtime-profile-name`
    - `--diffusion-enable-step-chunk`
    - `--diffusion-enable-chunk-preemption`
    - `--instance-scheduler-p95-fusion-tail-budget-ratio`
    - `--instance-scheduler-p95-fusion-heavy-threshold-s`
    - `--instance-scheduler-p95-fusion-min-chunk-steps`
    - `--instance-scheduler-p95-fusion-max-chunk-steps`

### 3.4 `slo_first` / `slack_age` / `slack_cost_age` / `slack_hybrid`

这四种是 deadline-aware 策略。最少需要：

1. 选择其中一种策略
2. 给请求提供 deadline 来源，或给实例提供静态默认 SLO

配置示例：

```bash
--instance-scheduler-policy slo_first \
--instance-scheduler-slo-target-ms 1800
```

如果没有任何显式 deadline 来源，请求会回退到 learned-p95 synthetic deadline：

```text
deadline_ts = arrival_time + dynamic_p95_ms / 1000
```

其中 `dynamic_p95_ms` 复用实例内 learned p95 + backlog 修正逻辑。也就是说，当前只提供 `estimated_cost_s` 的请求，仍然可以参与所有 slack 策略的 deadline 计算。

推荐同时配置：

- `--instance-scheduler-slo-floor-ms`
  - 对请求级或实例级 SLO 做下界保护
- `--instance-scheduler-aging-factor`
  - 影响 `best_effort` 队列老化排序
  - 同时也影响 `slack_age`、`slack_cost_age` 和 `slack_hybrid` 的老化项
- `--instance-scheduler-slack-panic-threshold`
  - `slack_hybrid` 的 Panic Mode 触发阈值 `δ`
- `--instance-scheduler-slack-swap-overhead-ms`
  - `slack_hybrid` 在 slack 分子中扣除的切换开销 `T_swap`
- `--instance-runtime-profile-path`
- `--instance-runtime-profile-name`
  - 提高 `estimated_cost_s` 估算质量
- `--instance-scheduler-p95-first-base-ms` / `min-ms` / `max-ms` / `backlog-alpha`
  - 当请求没有显式 deadline 时，这几项会直接影响 learned-p95 synthetic deadline 的松紧程度

## 4. deadline 来源与优先级

当前代码只从 `sampling_params.extra_args` 和实例配置读取 deadline，不读取 `OmniDiffusionRequest.deadline_ts` 字段。

deadline 计算优先级如下：

1. `sampling_params.extra_args["deadline_ts"]`
2. `sampling_params.extra_args["slo_target_ms"]`
3. `sampling_params.extra_args["slo_ms"]`
4. `instance_scheduler_slo_target_ms`
5. 如果以上都没有，则对 `slo_first`、`slack_age`、`slack_cost_age`、`slack_hybrid` 回退到 learned-p95 synthetic deadline；`p95-first-deadline` 则始终使用自己的 normalized synthetic deadline；其它策略仍可视为 `inf`

当命中 `slo_target_ms`、`slo_ms` 或实例级 `instance_scheduler_slo_target_ms` 时，实际计算方式为：

```text
deadline_ts = base_arrival_time + max(slo_target_ms, instance_scheduler_slo_floor_ms) / 1000
```

其中 `base_arrival_time` 的取值优先级为：

1. `request.arrival_time`
2. 当前入队时间

因此：

- 如果请求已经直接给出 `deadline_ts`，不会再应用 `instance_scheduler_slo_floor_ms`
- 如果只给出 `slo_ms`，当前实现会把它当作 `slo_target_ms` 的别名
- 如果请求没有显式 deadline，deadline-aware 策略会回退到 learned-p95 synthetic deadline
- synthetic deadline 同样锚定 `arrival_time`，相同请求多次 chunk 重新入队时不会重置

## 5. cost estimation 来源与优先级

当前代码只从 `sampling_params.extra_args` 注入显式耗时，不读取 `OmniDiffusionRequest.estimated_cost_s` 字段。

优先级：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

补充规则：

- 如果请求显式注入了 `estimated_cost_s`，调度器会按剩余 steps 比例缩放，得到“剩余耗时”
- profile 估算支持：
  - 单个 JSON 文件
  - 一个目录下的多个 JSON 文件
- profile 支持从 `profiles` 或 `entries` 字段读取记录
- 如果配置了 `instance_runtime_profile_name`，会按 JSON 中的 `instance_type` 过滤
- 若 profile 路径不存在、内容损坏或没有匹配记录，会自动回退到启发式估算

当前启发式大致依赖这些输入：

- `num_inference_steps`
- `executed_steps`
- `num_frames`
- `width` / `height` / `resolution`
- `num_outputs_per_prompt`

## 6. deadline-aware 策略的排序逻辑

`slo_first`、`slack_age`、`slack_cost_age` 不再完全共享同一套排序过程。

### 6.1 `slo_first` 的两阶段排序

`slo_first` 仍然保留先划分 `on_time` / `best_effort`，再分别排序的两阶段过程。调度器会：

1. 先按 deadline 从早到晚排序
2. 逐个把请求加入前缀集合
3. 如果当前前缀在预计可用时间之后已经无法全部按时完成，就从前缀中删掉“耗时最长”的那个请求

最终得到：

- `on_time_queue`
  - 预计还能按时完成的请求集合
- `best_effort_queue`
  - 当前看来已经来不及按时完成的请求集合

这里的预计可用时间还会考虑当前 active request 的剩余执行时间，因此该策略不是只看等待队列本身。

`slo_first` 的排序规则：

- `on_time_queue`
  - 按 `slack / remaining_cost` 从小到大
- `best_effort_queue`
  - 按 `remaining_cost_s / (1 + aging_factor * age)` 从小到大

### 6.2 `slack_age` / `slack_cost_age` 的单队列排序

这两个策略现在直接对整个等待队列做一次排序，不再先拆成 `on_time` 和 `best_effort` 两个集合。

- `slack_age`
  - 按 `slack - aging_factor * age` 从小到大
- `slack_cost_age`
  - 按 `slack + 0.25 * remaining_cost - aging_factor * age` 从小到大

其中：

- `slack = deadline_ts - now - remaining_cost_s`
- `age` 优先使用 `request.arrival_time` 计算

### 6.3 `slack_hybrid` 的 Panic / Throughput 双模式排序

`slack_hybrid` 先计算每个任务的 slack ratio：

- `slack_ratio = (deadline_ts - now - swap_overhead_s) / remaining_cost_s`
- 当请求没有显式 deadline 时，这里的 `deadline_ts` 使用 learned-p95 synthetic deadline

当任一等待任务，或当前 active request 的 slack ratio 小于 `panic_threshold` 时：

- 进入 `panic_edf` 模式
- 优先紧急任务（`slack_ratio < panic_threshold`）
- 紧急任务内部按 EDF 排序，tie-break 依次看 `slack_ratio`、`remaining_cost_s`、入队顺序

否则：

- 进入 `throughput_srpt` 模式
- 按 `remaining_cost_s - aging_factor * age_s` 排序

该策略为匹配 Panic Mode 的切换语义，会默认开启 step chunk 和 chunk preemption。

### 6.4 `p95-bucket-sjf` 的 deadline bucket + bucket 内 SJF 排序

这套策略的执行顺序可以直接拆成 5 步：

1. 对每个等待请求计算 `estimated_cost_s`。
2. 用实例内 learned p95 得到该请求的 `target_p95_ms`。
3. 派生 `deadline_ts` 和 `urgency_ms`。
4. 按 `urgency_ms` 落到有限个 bucket。
5. bucket 间按紧急程度排序，bucket 内按 SJF 排序。

### 6.4.1 先为每个请求生成 synthetic deadline

对于等待请求 `req`：

```text
estimated_cost_ms(req) = estimated_cost_s(req) * 1000
history_p95_ms = learned_p95_ms(instance)
target_p95_ms(req) = max(history_p95_ms, estimated_cost_ms(req))
deadline_ts(req) = arrival_ts(req) + target_p95_ms(req) / 1000
```

这里有两个关键语义：

- deadline 不是从客户端传进来的 contract，而是调度器内部为了排序派生出的 synthetic deadline。
- `target_p95_ms` 至少不小于请求自身成本，因此不会把明显的大请求错误地塞进“极短 deadline”。

### 6.4.2 再用实例可用时间计算紧迫度

当前实现不是简单拿 `deadline_ts - now` 作为紧迫度，而是扣除了 active request 的 total remaining cost：

```text
availability_ts = now + active_total_remaining_cost_s
urgency_ms(req) = (deadline_ts(req) - availability_ts) * 1000
```

因此：

- 如果实例前面已经堆了很长的 active work，等待请求的紧迫度会同步变小。
- 这使得 bucket 更接近“当前实例视角下的可救程度”，而不是纯 arrival-relative deadline。

### 6.4.3 用固定 bucket 数量做粗分层

当前代码使用：

- `instance_scheduler_p95_bucket_count`
- `instance_scheduler_p95_bucket_min_window_ms`

先求当前等待队列的锚定窗口：

```text
anchor_window_ms = max(history_p95_ms, max_estimated_cost_ms_in_queue, min_bucket_window_ms)
bucket_width_ms = anchor_window_ms / bucket_count
```

再把请求落桶：

```text
raw_bucket_id =
    0, if urgency_ms <= 0
    min(floor(urgency_ms / bucket_width_ms), bucket_count - 1), otherwise
```

这表示：

- 最紧急的请求进入 bucket `0`
- 越不紧急的请求，bucket id 越大
- 超出范围的请求统一压到最后一个 bucket，避免 bucket 数量无限膨胀

### 6.4.4 对老请求做有限 bucket promotion

若请求等待时间过长，可跨 bucket 上浮，但不是直接跳到队首：

```text
if age_s >= starvation_threshold_s:
    effective_bucket_id = max(raw_bucket_id - promote_levels, 0)
else:
    effective_bucket_id = raw_bucket_id
```

这样设计的原因是：

- 只修正 bucket 层级，不破坏 bucket 内的 SJF 语义
- 不把“防饥饿”重新写成难解释的分数项
- promotion 强度直接由 bucket 层数控制，可观测性更好

### 6.4.5 最终按“两级排序”出队

最终排序键是：

```text
(
    effective_bucket_id,
    estimated_cost_s,
    enqueue_time,
    sequence_id,
)
```

所以：

- 不同 bucket 之间，本质是 synthetic-deadline-aware priority queue
- 同一个 bucket 内，本质是 SJF
- 如果多个请求 bucket 和 cost 都相同，则保持更稳定的先来先服务顺序

这也是它和 `p95-first` 的主要区别：

- `p95-first`：在全队列上做单分数 greedy 选择
- `p95-bucket-sjf`：先粗分层，再层内 SJF

从工程角度看，后者更像“可解释的分层调度器”，前者更像“连续风险打分调度器”。

## 7. 当前输出的调度指标

所有请求都会带上的通用指标包括：

- `scheduler_policy`
- `queue_wait_ms`
- `scheduler_execute_ms`
- `scheduler_latency_ms`
- `queue_len`
- `dispatch_epoch`
- `chunk_budget_steps`
- `width`
- `height`
- `resolution`
- `num_frames`
- `total_steps`
- `executed_steps`
- `remaining_steps`
- `arrival_ts`
- `first_enqueue_ts`
- `first_dispatch_ts`
- `last_dispatch_ts`
- `last_preempted_ts`
- `completion_ts`
- `failure_ts`
- `aborted_ts`

`sjf` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`

`sjf_aging` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `age_s`
- `aging_factor`
- `aged_cost_s`
- `queue_rank`

`sjf_aging_guarded` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `age_s`
- `aging_factor`
- `aging_cost_weight`
- `aged_cost_s`
- `tail_protected`
- `protection_threshold_s`
- `wait_ratio`
- `learned_wait_guard_s`
- `dispatch_group`
- `queue_rank`

`sjf_aging_guarded_tail` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `age_s`
- `aging_factor`
- `aging_cost_weight`
- `aged_cost_s`
- `tail_protected`
- `tail_sunk`
- `super_heavy`
- `protection_threshold_s`
- `sink_threshold_s`
- `hard_escape_threshold_s`
- `wait_ratio`
- `learned_wait_guard_s`
- `queue_median_cost_s`
- `queue_p75_cost_s`
- `queue_p90_cost_s`
- `defer_relief_score`
- `defer_harm_score`
- `lighter_request_count`
- `lighter_mean_cost_s`
- `tail_defer_budget_ratio`
- `tail_defer_budget_limit`
- `global_arrived_unique`
- `global_budget_limit`
- `global_budget_used`
- `global_budget_remaining`
- `window_arrived_unique`
- `window_budget_limit`
- `window_budget_used`
- `window_budget_remaining`
- `dispatch_group`
- `queue_rank`

`p95-first` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `work_units`
- `estimated_service_ms`
- `learned_slowdown_p95`
- `observed_service_rate_ms_per_work_unit`
- `service_rate_source`
- `active_chunk_blocking_ms`
- `active_chunk_blocking_s`
- `predicted_finish_latency_ms`
- `target_latency_ms`
- `pressure_ratio`
- `risk_ms`
- `age_s`
- `starvation_boost`
- `queue_rank`

`p95-first-deadline` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `work_units`
- `estimated_service_ms`
- `learned_slowdown_p95`
- `observed_service_rate_ms_per_work_unit`
- `service_rate_source`
- `availability_ts`
- `active_chunk_blocking_ms`
- `active_chunk_blocking_s`
- `target_latency_ms`
- `synthetic_deadline_ts`
- `urgency_ms`
- `slack_s`
- `pressure_ratio`
- `queue_rank`

`p95-bucket-sjf` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `history_p95_ms`
- `target_p95_ms`
- `deadline_ts`
- `urgency_ms`
- `raw_bucket_id`
- `effective_bucket_id`
- `bucket_width_ms`
- `age_s`
- `starvation_promoted`
- `queue_rank`

`p95-bucket-sjf-normalized` 额外会输出：

- `queue_reorder_count`
- `learned_slowdown_p95`
- `observed_service_rate_ms_per_work_unit`
- `service_rate_source`
- `estimated_cost_s`
- `work_units`
- `estimated_service_ms`
- `target_latency_ms`
- `synthetic_deadline_ts`
- `urgency_ms`
- `raw_bucket_id`
- `effective_bucket_id`
- `bucket_width_ms`
- `age_s`
- `starvation_promoted`
- `queue_rank`

`p95-fusion` 额外会输出：

- `queue_reorder_count`
- `estimated_cost_s`
- `estimated_service_ms`
- `estimated_service_s`
- `dynamic_p95_ms`
- `learned_p95_ms`
- `backlog_adjusted_p95_ms`
- `learned_slowdown_p95`
- `backlog_s_at_schedule`
- `instance_backlog_total_s`
- `active_chunk_blocking_ms`
- `active_chunk_blocking_s`
- `target_latency_ms`
- `predicted_finish_latency_ms`
- `pressure_ratio`
- `slack_s`
- `slack_ratio`
- `age_s`
- `is_heavy`
- `is_urgent`
- `tail_candidate`
- `tail_lane_selected`
- `tail_lane_rank`
- `scheduler_force_run_to_completion`
- `scheduler_chunk_budget_steps`
- `nonheavy_streak_seed`
- `tail_budget_ratio`
- `ratio_cap`
- `borrowed_cap`
- `lane_cap`
- `nonheavy_streak_limit`
- `dispatch_group`
- `queue_rank`

`slack_hybrid` 额外会输出：

- `hybrid_mode`
- `slack_ratio`
- `panic_threshold`
- `swap_overhead_ms`
- `throughput_priority`
- `priority_score`
- `is_urgent`
- `active_slack_ratio`
- `queue_rank`

deadline-aware 策略额外会输出：

- `attain_before`
- `attain_after`
- `self_hit`
- `damage_count`
- `on_time_set_size`
- `best_effort_set_size`
- `tail_set_size`
- `regret_drop_count`
- `queue_reorder_count`
- `deadline_slack_ms`
- `dispatch_group`
- `estimated_cost_s`

指标含义：

- `attain_before`
  - 新请求加入前，旧等待集合中可按时完成的请求数
- `attain_after`
  - 新请求加入后，当前等待集合中可按时完成的请求数
- `self_hit`
  - 新请求自己是否进入 `on_time` 集合，取值 `0/1`
- `damage_count`
  - 新请求加入后，从原 `on_time` 集合掉出的旧请求数
- `dispatch_group`
  - 新请求最终被归到 `on_time` 或 `best_effort`
- `deadline_slack_ms`
  - 按当前计划顺序推算出的完成时刻距离 deadline 的余量；若 deadline 为 `inf`，则为 `None`

## 8. 配置示例

### 8.1 最小 `sjf` 配置

```bash
--instance-scheduler-policy sjf
```

若请求侧能注入更准确的估时：

```text
sampling_params.extra_args["estimated_cost_s"] = 1.8
```

### 8.2 推荐的 `p95-first` 配置

```bash
--instance-scheduler-policy p95-first \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption \
--diffusion-chunk-budget-steps 4
```

如果需要温和的防饥饿保护，可额外配置：

```bash
--instance-scheduler-p95-first-age-bias 0.1 \
--instance-scheduler-p95-first-starvation-threshold-s 8 \
--instance-scheduler-p95-first-starvation-boost 0.2
```

请求侧最小仍建议提供：

```text
sampling_params.extra_args["estimated_cost_s"] = 0.9
```

### 8.3 推荐的 `p95-first-deadline` 配置

```bash
--instance-scheduler-policy p95-first-deadline \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption \
--diffusion-chunk-budget-steps 4
```

请求侧最小仍建议提供：

```text
sampling_params.extra_args["estimated_cost_s"] = 0.9
```

### 8.4 推荐的 `p95-bucket-sjf` 配置

```bash
--instance-scheduler-policy p95-bucket-sjf \
--instance-scheduler-p95-bucket-count 4 \
--instance-scheduler-p95-bucket-min-window-ms 250 \
--instance-scheduler-p95-bucket-starvation-threshold-s 8 \
--instance-scheduler-p95-bucket-starvation-promote-levels 1 \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption
```

请求侧最小只需要：

```text
sampling_params.extra_args["estimated_cost_s"] = 0.9
```

### 8.5 推荐的 `p95-bucket-sjf-normalized` 配置

```bash
--instance-scheduler-policy p95-bucket-sjf-normalized \
--instance-scheduler-p95-bucket-count 4 \
--instance-scheduler-p95-bucket-min-window-ms 250 \
--instance-scheduler-p95-bucket-starvation-threshold-s 8 \
--instance-scheduler-p95-bucket-starvation-promote-levels 1 \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption
```

### 8.6 推荐的 `sjf_aging_guarded_tail` 配置

```bash
--instance-scheduler-policy sjf_aging_guarded_tail \
--instance-scheduler-aging-factor 0.25 \
--instance-scheduler-sjf-aging-guarded-tail-defer-budget-ratio 0.02 \
--instance-scheduler-sjf-aging-guarded-tail-hard-escape-wait-multiplier 100.0 \
--instance-scheduler-sjf-aging-guarded-tail-hard-escape-cost-multiplier 100.0 \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption \
--diffusion-chunk-budget-steps 4
```

这条策略当前建议显式带上 3 个专属参数：一个把沉降预算压到明显低于 `p95` 边界，两个把 `hard_escape` 阈值抬高到默认近似关闭。默认实现仍保留：

- learned wait guard
- queue-local `super_heavy` 判定
- `defer_relief_score` 候选排序
- 很高的 hard escape 阈值

共同限制沉降行为。

### 8.7 推荐的 `p95-fusion` 配置

```bash
--instance-scheduler-policy p95-fusion \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--instance-scheduler-p95-fusion-tail-budget-ratio 0.10 \
--instance-scheduler-p95-fusion-heavy-threshold-s 20 \
--instance-scheduler-p95-fusion-urgent-slack-ratio 1.0 \
--instance-scheduler-p95-fusion-promote-wait-s 60 \
--instance-scheduler-p95-fusion-nonheavy-streak-limit 4 \
--instance-scheduler-p95-fusion-growth-every 20 \
--instance-scheduler-p95-fusion-borrowed-cap-max 4 \
--instance-scheduler-p95-fusion-min-chunk-steps 1 \
--instance-scheduler-p95-fusion-max-chunk-steps 8 \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption
```

如果你希望更积极地保护 heavy 请求，可以优先调这几项：

```bash
--instance-scheduler-p95-fusion-tail-budget-ratio 0.15 \
--instance-scheduler-p95-fusion-promote-wait-s 45 \
--instance-scheduler-p95-fusion-nonheavy-streak-limit 3
```

### 8.8 推荐的 deadline-aware 配置

```bash
--instance-scheduler-policy slack_age \
--instance-scheduler-slo-target-ms 1800 \
--instance-scheduler-slo-floor-ms 800 \
--instance-scheduler-aging-factor 0.25 \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a \
--diffusion-enable-step-chunk \
--diffusion-enable-chunk-preemption \
--diffusion-chunk-budget-steps 4
```

### 8.9 请求级覆盖示例

显式给绝对 deadline：

```text
sampling_params.extra_args["deadline_ts"] = 1710000000.0
```

显式给相对 SLO：

```text
sampling_params.extra_args["slo_target_ms"] = 2500
sampling_params.extra_args["estimated_cost_s"] = 0.9
```

兼容旧字段名：

```text
sampling_params.extra_args["slo_ms"] = 2500
```

## 9. 参数合法性约束

当前 `OmniDiffusionConfig` 校验要求：

- `instance_scheduler_policy` 必须属于
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `bypass_guard_sjf`
  - `size_bucket_sjf_aging`
  - `type_fifo_defer_budget`
  - `slo_first`
  - `p95-first`
  - `p95-first-deadline`
  - `p95-bucket-sjf`
  - `p95-bucket-sjf-normalized`
  - `p95-fusion`
  - `slack_age`
  - `slack_cost_age`
  - `slack_hybrid`
- `instance_scheduler_slo_target_ms` 如果设置，必须 `> 0`
- `instance_scheduler_slo_floor_ms >= 0`
- `instance_scheduler_aging_factor >= 0`
- `instance_scheduler_p95_first_base_ms > 0`（若设置）
- `instance_scheduler_p95_first_min_ms >= 0`
- `instance_scheduler_p95_first_max_ms > 0`（若设置），且不得小于 `instance_scheduler_p95_first_min_ms`
- `instance_scheduler_p95_first_backlog_alpha >= 0`
- `instance_scheduler_p95_first_size_bias >= 0`
- `instance_scheduler_p95_first_age_bias >= 0`
- `instance_scheduler_p95_first_starvation_threshold_s > 0`（若设置）
- `instance_scheduler_p95_first_starvation_boost >= 0`
- `instance_scheduler_p95_bucket_count >= 1`
- `instance_scheduler_p95_bucket_min_window_ms > 0`
- `instance_scheduler_p95_bucket_starvation_threshold_s > 0`（若设置）
- `instance_scheduler_p95_bucket_starvation_promote_levels >= 0`
- `instance_scheduler_slack_panic_threshold >= 0`
- `instance_scheduler_slack_swap_overhead_ms >= 0`
- `instance_scheduler_type_fifo_defer_budget_ratio` 必须在 `[0, 1]`
- `instance_scheduler_type_fifo_defer_hard_escape_wait_multiplier > 0`
- `instance_scheduler_type_fifo_defer_hard_escape_cost_multiplier > 0`
- `instance_scheduler_p95_fusion_tail_budget_ratio` 必须在 `(0, 1]`
- `instance_scheduler_p95_fusion_heavy_threshold_s > 0`
- `instance_scheduler_p95_fusion_urgent_slack_ratio >= 0`
- `instance_scheduler_p95_fusion_promote_wait_s > 0`
- `instance_scheduler_p95_fusion_nonheavy_streak_limit >= 1`
- `instance_scheduler_p95_fusion_growth_every >= 1`
- `instance_scheduler_p95_fusion_borrowed_cap_max >= 0`
- `instance_scheduler_p95_fusion_min_chunk_steps >= 1`
- `instance_scheduler_p95_fusion_max_chunk_steps >= instance_scheduler_p95_fusion_min_chunk_steps`
- 当策略为 `p95-first`、`p95-first-deadline`、`p95-bucket-sjf`、`p95-bucket-sjf-normalized`、`p95-fusion`、`slack_hybrid`、`sjf_aging_guarded`、`sjf_aging_guarded_tail`、`type_fifo_defer_budget` 时，会自动启用：
  - `diffusion_enable_step_chunk=True`
  - `diffusion_enable_chunk_preemption=True`
- `diffusion_enable_chunk_preemption=True` 时，`diffusion_enable_step_chunk` 必须为 `True`
- `diffusion_chunk_budget_steps >= 1`
- `diffusion_image_chunk_budget_steps >= 1`（若设置）
- `diffusion_video_chunk_budget_steps >= 1`（若设置）
- `diffusion_small_request_latency_threshold_ms > 0`（若设置）

## 10. 当前仓库里的验证证据

从当前代码和测试可以确认：

- 配置入口已接到 `AsyncOmni` 和 CLI
- 17 种策略都已接入配置校验
- `sjf` 已覆盖：
  - 队列重排
  - remaining steps 缩放
  - runtime profile 估时
- deadline-aware 策略已覆盖：
  - `on_time` / `best_effort` 集合划分
  - `slo_first`、`slack_age`、`slack_cost_age` 的排序差异
  - `attain_before`、`self_hit` 等指标输出
- `sjf_aging_guarded_tail` 已接入 CLI、配置与调度实现
- `sjf_aging_guarded_tail` 当前已有针对性测试覆盖：
  - old super-heavy request 会被沉降到队尾
  - 小样本下也能借助 bootstrap slot 触发沉降
  - 已经 sunk 的 waiting request 在后续 reorder 中仍保持 sunk
  - 2% 预算耗尽后，不会继续沉降新的请求
  - 只有命中 hard escape 的非 sunk 请求才会触发 guarded run-to-completion
- `p95-first`、`p95-first-deadline`、`p95-bucket-sjf`、`p95-bucket-sjf-normalized`、`p95-fusion`、`slack_hybrid` 已接入 CLI、配置与调度实现
- `p95-fusion` 当前已有针对性测试覆盖：
  - CLI / 默认 stage config 参数透传
  - 自动启用 `step chunk` 与 `chunk preemption`
  - heavy 请求轮转进入保护顺序
  - overdue heavy 切换 `run_to_completion`
  - heavy 请求即使未进入 `tail_lane` 也能拿到 chunk override
  - scheduler 空闲时重置 `nonheavy_streak`

这份文档只描述当前仓库行为，不再保留“统一采用 `3 x estimated_cost_s` 生成 SLO”这类实验约定。若要使用该规则，应在请求构造侧显式写入 `sampling_params.extra_args["slo_target_ms"]`，而不是假定调度器内部会自动生成。
