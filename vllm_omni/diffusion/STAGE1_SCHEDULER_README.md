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

`OmniDiffusionConfig.instance_scheduler_policy` 当前支持以下 5 种取值：

- `fcfs`
- `sjf`
- `slo_first`
- `slack_age`
- `slack_cost_age`

CLI 入口：

```bash
--instance-scheduler-policy {fcfs,sjf,slo_first,slack_age,slack_cost_age}
```

其中：

- `fcfs`
  - 严格按入队顺序调度
- `sjf`
  - 按估算剩余耗时从小到大排序
- `slo_first`
  - 先求可按时完成的 `on_time` 集合，再按 `slack / remaining_cost` 排序
- `slack_age`
  - 与 `slo_first` 使用同一套 deadline-aware 分组逻辑，但 `on_time` 集合内改为按 `slack - aging_factor * age` 排序
- `slack_cost_age`
  - 与 `slack_age` 类似，但会额外加入一个有限的剩余耗时项，分数为 `slack + 0.25 * remaining_cost - aging_factor * age`

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

### 3.2 `sjf`

不要求提供 deadline，但强烈建议提供更准确的耗时估计来源。当前耗时估计优先级是：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

### 3.3 `slo_first` / `slack_age` / `slack_cost_age`

这三种是 deadline-aware 策略。最少需要：

1. 选择其中一种策略
2. 给请求提供 deadline 来源，或给实例提供静态默认 SLO

配置示例：

```bash
--instance-scheduler-policy slo_first \
--instance-scheduler-slo-target-ms 1800
```

如果没有任何 deadline 来源，代码会把 deadline 视为 `inf`。策略仍能运行，但会退化为“没有实际 deadline 约束的排序”。

推荐同时配置：

- `--instance-scheduler-slo-floor-ms`
  - 对请求级或实例级 SLO 做下界保护
- `--instance-scheduler-aging-factor`
  - 影响 `best_effort` 队列老化排序
  - 同时也影响 `slack_age` 和 `slack_cost_age` 的 `on_time` 排序
- `--instance-runtime-profile-path`
- `--instance-runtime-profile-name`
  - 提高 `estimated_cost_s` 估算质量

## 4. deadline 来源与优先级

当前代码只从 `sampling_params.extra_args` 和实例配置读取 deadline，不读取 `OmniDiffusionRequest.deadline_ts` 字段。

deadline 计算优先级如下：

1. `sampling_params.extra_args["deadline_ts"]`
2. `sampling_params.extra_args["slo_target_ms"]`
3. `sampling_params.extra_args["slo_ms"]`
4. `instance_scheduler_slo_target_ms`
5. 如果以上都没有，则为 `inf`

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
- 相同请求多次 chunk 重新入队时，deadline 仍然锚定首次 arrival，而不是最近一次 requeue 时间

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

`slo_first`、`slack_age`、`slack_cost_age` 共享同一个两阶段过程。

### 6.1 先划分 `on_time` 和 `best_effort`

调度器会：

1. 先按 deadline 从早到晚排序
2. 逐个把请求加入前缀集合
3. 如果当前前缀在预计可用时间之后已经无法全部按时完成，就从前缀中删掉“耗时最长”的那个请求

最终得到：

- `on_time_queue`
  - 预计还能按时完成的请求集合
- `best_effort_queue`
  - 当前看来已经来不及按时完成的请求集合

这里的预计可用时间还会考虑当前 active request 的剩余执行时间，因此该策略不是只看等待队列本身。

### 6.2 再分别给两个集合排序

`on_time_queue`：

- `slo_first`
  - 按 `slack / remaining_cost` 从小到大
- `slack_age`
  - 按 `slack - aging_factor * age` 从小到大
- `slack_cost_age`
  - 按 `slack + 0.25 * remaining_cost - aging_factor * age` 从小到大

其中：

- `slack = deadline_ts - now - remaining_cost_s`
- `age` 优先使用 `request.arrival_time` 计算

`best_effort_queue`：

- 三种 deadline-aware 策略都统一按
  - `remaining_cost_s / (1 + aging_factor * age)`
  - 从小到大排序

因此：

- `aging_factor=0` 时，`best_effort` 集合更像“短作业优先”
- `aging_factor>0` 时，老请求会逐渐前移

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

### 8.2 推荐的 deadline-aware 配置

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

### 8.3 请求级覆盖示例

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
