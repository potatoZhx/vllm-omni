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

`OmniDiffusionConfig.instance_scheduler_policy` 当前支持以下 9 种取值：

- `fcfs`
- `sjf`
- `sjf_aging`
- `slo_first`
- `p95-first`
- `p95-bucket-sjf`
- `slack_age`
- `slack_cost_age`
- `slack_hybrid`

CLI 入口：

```bash
--instance-scheduler-policy {fcfs,sjf,sjf_aging,slo_first,p95-first,p95-bucket-sjf,slack_age,slack_cost_age,slack_hybrid}
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

### 1.1 各策略可调参数与默认值

| 策略 | 主要排序逻辑 | 可调参数 | 默认值 |
| --- | --- | --- | --- |
| `fcfs` | 按 `enqueue_time` 先来先服务 | 无 | 无 |
| `sjf` | 按 `estimated_cost_s` 从小到大排序 | 无 | 无 |
| `sjf_aging` | 按 `estimated_cost_s / (1 + aging_factor * age_s)` 排序 | `instance_scheduler_aging_factor` | `0.0`<br>实现中当 `<= 0` 时实际回退为内建 aging 因子 `1.0` |
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

### 3.2 `sjf` / `sjf_aging`

这两种策略都不要求提供 deadline，但强烈建议提供更准确的耗时估计来源。当前耗时估计优先级是：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

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
