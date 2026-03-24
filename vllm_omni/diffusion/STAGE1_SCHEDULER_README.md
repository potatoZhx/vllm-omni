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

`OmniDiffusionConfig.instance_scheduler_policy` 当前支持以下 10 种取值：

- `fcfs`
- `sjf`
- `sjf_aging`
- `size_bucket_sjf_aging`
- `slo_first`
- `p95-first`
- `p95-bucket-sjf`
- `slack_age`
- `slack_cost_age`
- `slack_hybrid`

CLI 入口：

```bash
--instance-scheduler-policy {fcfs,sjf,sjf_aging,size_bucket_sjf_aging,slo_first,p95-first,p95-bucket-sjf,slack_age,slack_cost_age,slack_hybrid}
```

其中：

- `fcfs`
  - 严格按入队顺序调度
- `sjf`
  - 按估算剩余耗时从小到大排序
- `sjf_aging`
  - 在 `sjf` 的剩余耗时排序上叠加等待时长老化分数 `remaining_cost / (1 + aging_factor * age)`
  - 当 `instance_scheduler_aging_factor <= 0` 时，使用内建默认 aging 系数 `1.0`，保证不额外配参也具备防饥饿能力
  - 对 step chunk / chunk preemption 友好：chunk 重入队后继续按“剩余耗时 + 首次 arrival 老化”计算
- `size_bucket_sjf_aging`
  - 先按固定分辨率 bucket 分组，再在 bucket 内按 `remaining_cost / (1 + aging_factor * age)` 做 SJF + Aging 排序
  - aging 会按等待时间逐步提升大 bucket 请求，避免长期饥饿
- `slo_first`
  - 先求可按时完成的 `on_time` 集合，再按 `slack / remaining_cost` 排序
- `slack_age`
  - 直接对整个等待队列按 `slack - aging_factor * age` 做单队列排序
- `slack_cost_age`
  - 直接对整个等待队列按 `slack + 0.25 * remaining_cost - aging_factor * age` 做单队列排序
- `slack_hybrid`
  - 基于松弛度比值 `((deadline - now - swap_overhead) / T_rem)` 在两种模式之间切换
  - 若任一任务的 slack ratio 小于 `panic_threshold`，进入 Panic Mode，按 EDF 调度并优先紧急任务
  - 否则进入 Throughput Mode，按 `T_rem - aging_factor * wait_time` 做 SRPT + Aging 排序

### 1.0 `p95-first` 算法简介

`p95-first` 不是平均时延最小化策略，而是一个“优先压低实例内尾时延”的单队列评分调度器。

它的核心思路是：

- 先为当前实例维护一个动态目标 `dynamic_p95_ms`
- 再估计“如果某个请求现在被选中，它最终大概会在什么时候完成”
- 最后计算这个请求距离动态 p95 目标还有多危险，并按危险度排序

当前实现里的关键量如下：

- `remaining_cost_s`
  - 请求的剩余成本
  - 若请求显式提供 `sampling_params.extra_args["estimated_cost_s"]`，则调度器和 engine 都按剩余 steps 比例缩放这份 cost
  - 否则回退到 runtime profile 或启发式估算
- `dynamic_p95_ms`
  - 动态尾时延目标
  - 由在线学习得到的 `learned_p95_ms`、静态基线 `base_ms`、最小值 `min_ms`、以及 backlog 修正共同决定，并受 `max_ms` 限幅
- `predicted_finish_latency_ms`
  - 预测若当前请求此刻被调度，它完成时的总 latency
  - 近似为：

```text
predicted_finish_latency_ms = (age_s + cursor_s + remaining_cost_s) * 1000
```

- `risk_ms`
  - 预测完成时延相对动态 p95 目标的超额

```text
risk_ms = predicted_finish_latency_ms - dynamic_p95_ms
```

- `final_priority_score`
  - 当前实现的最终排序分数，按分数从高到低选择下一请求
  - 基础项是“单位成本上的风险压力”，再叠加 size bias、age bias 和 starvation boost

```text
base_score = -risk_ms / max(remaining_cost_s * 1000, 1e-9)
final_priority_score = base_score + size_bias * remaining_cost_s - age_bias * age_s - starvation_boost
```

直观理解：

- 一个请求如果预计会明显冲破当前 `dynamic_p95_ms`，它的风险会变大
- 但如果它本身非常长，调度器不会简单粗暴地永远先救它，而是看“单位成本带来的 p95 风险改善值”
- 因此 `p95-first` 介于纯 deadline-first 和纯 SJF 之间，本质上是在做“尾延迟风险 / 剩余成本”的折中

与运行时机制的关系：

- `p95-first` 当前依赖 step chunk + chunk preemption 才能持续重排队列
- active request 不会在 chunk 内被打断，只会在 chunk 边界重入队后重新参与评分
- 因此该策略不是连续抢占模型，而是“chunk 边界上的风险重排”

适用边界：

- 该策略更适合混合长短任务、且希望压低实例内尾时延的场景
- 如果系统已经长期过载，`dynamic_p95_ms` 会被 backlog 拉高，此时排序分辨率会下降
- 如果请求没有显式 `estimated_cost_s`，最终效果会明显依赖 runtime profile 质量

### 1.1 `p95-bucket-sjf` 算法完整说明

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

### 1.2 各策略可调参数与默认值

| 策略 | 主要排序逻辑 | 可调参数 | 默认值 |
| --- | --- | --- | --- |
| `fcfs` | 按 `enqueue_time` 先来先服务 | 无 | 无 |
| `sjf` | 按 `estimated_cost_s` 从小到大排序 | 无 | 无 |
| `sjf_aging` | 按 `estimated_cost_s / (1 + aging_factor * age_s)` 排序 | `instance_scheduler_aging_factor` | `0.0`<br>实现中当 `<= 0` 时实际回退为内建 aging 因子 `1.0` |
| `size_bucket_sjf_aging` | 先按固定分辨率 bucket 排序，再在 bucket 内按 `estimated_cost_s / (1 + aging_factor * age_s)` 排序；等待过久时允许跨 bucket 晋升 | `instance_scheduler_aging_factor` | `0.0`<br>实现中当 `<= 0` 时实际回退为内建 aging 因子 `1.0` |
| `slo_first` | 先求 `on_time` 集合，再按 `slack / remaining_cost` 排序；尾部按 aging best-effort 排序 | `instance_scheduler_slo_target_ms`<br>`instance_scheduler_slo_floor_ms`<br>`instance_scheduler_aging_factor` | `None`<br>`0.0`<br>`0.0` |
| `p95-first` | 基于 `dynamic_p95_ms`、`risk_ms`、size bias、age bias、starvation boost 的单队列评分排序 | `instance_scheduler_p95_first_base_ms`<br>`instance_scheduler_p95_first_min_ms`<br>`instance_scheduler_p95_first_max_ms`<br>`instance_scheduler_p95_first_backlog_alpha`<br>`instance_scheduler_p95_first_size_bias`<br>`instance_scheduler_p95_first_age_bias`<br>`instance_scheduler_p95_first_starvation_threshold_s`<br>`instance_scheduler_p95_first_starvation_boost` | `None`<br>`0.0`<br>`None`<br>`1.0`<br>`0.0`<br>`0.0`<br>`None`<br>`0.0` |
| `p95-bucket-sjf` | 先按 `urgency_ms` 划 bucket，再在 bucket 内按 `estimated_cost_s` 做 SJF | `instance_scheduler_p95_bucket_count`<br>`instance_scheduler_p95_bucket_min_window_ms`<br>`instance_scheduler_p95_bucket_starvation_threshold_s`<br>`instance_scheduler_p95_bucket_starvation_promote_levels`<br>`instance_scheduler_p95_first_min_ms` | `4`<br>`200.0`<br>`None`<br>`1`<br>`0.0` |
| `slack_age` | 单队列按 `slack - aging_factor * age_s` 排序 | `instance_scheduler_aging_factor` | `0.0` |
| `slack_cost_age` | 单队列按 `slack + 0.25 * remaining_cost_s - aging_factor * age_s` 排序 | `instance_scheduler_aging_factor` | `0.0`<br>其中 `0.25` 是当前代码内常量，不是配置项 |
| `slack_hybrid` | 若任一请求 `slack_ratio < panic_threshold`，切到 Panic EDF；否则按 `estimated_cost_s - aging_factor * age_s` 做 Throughput SRPT + Aging 排序 | `instance_scheduler_aging_factor`<br>`instance_scheduler_slack_panic_threshold`<br>`instance_scheduler_slack_swap_overhead_ms` | `0.0`<br>`1.0`<br>`0.0` |

补充：

- `estimated_cost_s` 不是实例级配置，而是请求运行时输入；若请求未提供，调度器会回退到 runtime profile 或启发式估算。
- `deadline_ts`、`slo_target_ms`、`slo_ms` 也是请求运行时输入；若 deadline-aware 策略没有显式 deadline，会回退到 learned-p95 synthetic deadline。
- `p95-first`、`p95-bucket-sjf`、`slack_hybrid` 在当前实现里会自动启用：
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

### 3.2 `sjf` / `sjf_aging` / `size_bucket_sjf_aging`

这三种策略都不要求提供 deadline，但强烈建议提供更准确的耗时估计来源。当前耗时估计优先级是：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

其中 `size_bucket_sjf_aging` 额外会先按固定分辨率 bucket 分组：

- `max(width, height) <= 512`
- `512 < max(width, height) <= 768`
- `768 < max(width, height) <= 1024`
- `max(width, height) > 1024`

然后在 bucket 内做 `SJF + Aging`，并按等待时长逐步把大 bucket 请求往前提升。

### 3.3 `p95-first` / `p95-bucket-sjf`

这两种策略都只要求请求侧尽量提供更准确的 `estimated_cost_s`，不要求请求显式提供 `slo_ms`、`slo_target_ms` 或 `deadline_ts`。

其中：

- `p95-first`
  - 使用动态 p95 单队列评分排序
- `p95-bucket-sjf`
  - 本地学习历史 `history_p95_ms`
  - 对每个请求计算 `target_p95_ms = max(history_p95_ms, estimated_cost_ms)`
  - 用 `deadline_ts = arrival_ts + target_p95_ms / 1000` 派生内部 deadline
  - 按 `urgency_ms = (deadline_ts - availability_ts) * 1000` 做 bucket 划分
  - bucket 内按 `estimated_cost_s` 做 SJF 排序

推荐同时配置：

- `--instance-scheduler-p95-first-base-ms`
- `--instance-scheduler-p95-first-min-ms`
- `--instance-runtime-profile-path`
- `--instance-runtime-profile-name`
- `--diffusion-enable-step-chunk`
- `--diffusion-enable-chunk-preemption`

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
5. 如果以上都没有，则对 `slo_first`、`slack_age`、`slack_cost_age`、`slack_hybrid` 回退到 learned-p95 synthetic deadline；其它策略仍可视为 `inf`

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

### 8.2 推荐的 `p95-bucket-sjf` 配置

```bash
--instance-scheduler-policy p95-bucket-sjf \
--instance-scheduler-p95-first-base-ms 2500 \
--instance-scheduler-p95-first-min-ms 1200 \
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

### 8.3 推荐的 deadline-aware 配置

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

### 8.4 请求级覆盖示例

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
  - `slo_first`
  - `slack_age`
  - `slack_cost_age`
- `instance_scheduler_slo_target_ms` 如果设置，必须 `> 0`
- `instance_scheduler_slo_floor_ms >= 0`
- `instance_scheduler_aging_factor >= 0`
- `diffusion_enable_chunk_preemption=True` 时，`diffusion_enable_step_chunk` 必须为 `True`
- `diffusion_chunk_budget_steps >= 1`
- `diffusion_image_chunk_budget_steps >= 1`（若设置）
- `diffusion_video_chunk_budget_steps >= 1`（若设置）
- `diffusion_small_request_latency_threshold_ms > 0`（若设置）

## 10. 当前仓库里的验证证据

从当前代码和测试可以确认：

- 配置入口已接到 `AsyncOmni` 和 CLI
- 5 种策略都通过了配置校验
- `sjf` 已覆盖：
  - 队列重排
  - remaining steps 缩放
  - runtime profile 估时
- deadline-aware 策略已覆盖：
  - `on_time` / `best_effort` 集合划分
  - `slo_first`、`slack_age`、`slack_cost_age` 的排序差异
  - `attain_before`、`self_hit` 等指标输出

这份文档只描述当前仓库行为，不再保留“统一采用 `3 x estimated_cost_s` 生成 SLO”这类实验约定。若要使用该规则，应在请求构造侧显式写入 `sampling_params.extra_args["slo_target_ms"]`，而不是假定调度器内部会自动生成。
