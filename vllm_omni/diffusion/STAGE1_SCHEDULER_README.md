# Stage1 Scheduler README

本文档说明当前 `vllm_omni/diffusion/stage1_scheduler.py` 中 `slo_first` 策略的实际配置要求、参数优先级，以及当前实验阶段的临时 SLO 规则。

## 1. 当前支持的策略

`Stage1Scheduler` 当前支持：

- `fcfs`
- `sjf`
- `slo_first`

启用 `slo_first` 时，需要通过配置设置：

```text
instance_scheduler_policy = "slo_first"
```

## 2. 启用 `slo_first` 需要设置哪些参数

### 2.1 必配项

最少需要两类信息：

1. 打开策略：

```text
instance_scheduler_policy = "slo_first"
```

2. 提供 deadline 来源：

- 实例级默认：
  - `instance_scheduler_slo_target_ms`
- 或请求级覆盖：
  - `sampling_params.extra_args["deadline_ts"]`
  - `sampling_params.extra_args["slo_target_ms"]`
  - `sampling_params.extra_args["slo_ms"]`

如果没有任何 deadline 来源，当前实现会把 deadline 视为 `inf`。算法仍能运行，但不会形成真正的 SLO 驱动排序。

### 2.2 推荐项

建议同时配置：

- `instance_scheduler_slo_floor_ms`
  - 对目标 SLO 做下界保护
- `instance_scheduler_aging_factor`
  - 控制 tail set 的 aging，避免老请求长期饥饿
- `instance_runtime_profile_path`
  - 提供 runtime profile 作为 cost estimation 输入
- `instance_runtime_profile_name`
  - 当 profile 文件内有多种实例画像时，用于选择对应实例类型

## 3. 请求级参数优先级

当前代码里的 deadline 计算优先级是：

1. `sampling_params.extra_args["deadline_ts"]`
2. `sampling_params.extra_args["slo_target_ms"]`
3. `sampling_params.extra_args["slo_ms"]`
4. `instance_scheduler_slo_target_ms`
5. 如果以上都没有，则 deadline 为 `inf`

因此：

- 如果请求已经显式带了 `deadline_ts`，实例级 `instance_scheduler_slo_target_ms` 只作为兜底
- 如果 trace 里只有 `slo_ms`，当前实现也会把它作为 deadline 来源

## 4. cost estimation 参数优先级

`slo_first` 的排序依赖 `estimated_cost_s`。

当前实现的 cost estimation 优先级是：

1. `sampling_params.extra_args["estimated_cost_s"]`
2. runtime profile 估算
3. 启发式估算

如果没有请求级 `estimated_cost_s`，建议至少配置 runtime profile，否则排序质量会更依赖启发式估算。

## 5. 当前实验阶段的临时 SLO 规则

目前数据集里还没有每个请求单独的 SLO 标注。

因此当前实验阶段采用如下临时规则：

```text
slo_target_ms = 3 * estimated_cost_s * 1000
```

也就是说：

- 如果请求没有显式提供 `deadline_ts`
- 也没有显式提供 `slo_target_ms` 或 `slo_ms`
- 那么建议在请求构造阶段，按 `3 x estimated_cost_s` 生成请求级 `slo_target_ms`

建议采用以下优先顺序：

1. 若请求显式提供 `deadline_ts`
   - 直接使用
2. 否则若请求显式提供 `slo_target_ms` 或 `slo_ms`
   - 直接使用
3. 否则
   - 使用 `3 * estimated_cost_s * 1000` 生成 `slo_target_ms`

这个规则的定位是：

- 为当前 `slo_first` 算法验证提供统一、可运行的 deadline 来源
- 不等价于最终线上业务 SLO

## 6. 推荐配置示例

### 6.1 最小可用配置

```bash
--instance-scheduler-policy slo_first \
--instance-scheduler-slo-target-ms 1800
```

### 6.2 推荐配置

```bash
--instance-scheduler-policy slo_first \
--instance-scheduler-slo-target-ms 1800 \
--instance-scheduler-slo-floor-ms 800 \
--instance-scheduler-aging-factor 0.25 \
--instance-runtime-profile-path /profile/runtime.json \
--instance-runtime-profile-name img-a
```

### 6.3 使用临时 `3x estimated_cost_s` 规则时

如果你不想依赖固定实例级 `instance_scheduler_slo_target_ms`，也可以在请求构造阶段直接写入：

```text
sampling_params.extra_args["slo_target_ms"] = 3 * estimated_cost_s * 1000
```

## 7. 参数合法性约束

当前代码要求：

- `instance_scheduler_policy` 必须是 `fcfs`、`sjf`、`slo_first` 之一
- `instance_scheduler_slo_target_ms` 如果设置，必须 `> 0`
- `instance_scheduler_slo_floor_ms >= 0`
- `instance_scheduler_aging_factor >= 0`

## 8. 当前验证状态

当前分支里：

- `sjf` 已做过验证
- `slo_first` 的代码、配置入口和单测已经落地
- `slo_first` 还没有完成端到端实测验证

因此当前重点不是再补参数定义，而是基于上面的配置口径去做真实负载实验，确认：

- `attain_before`
- `attain_after`
- `self_hit`
- `damage_count`
- `deadline_slack_ms`

这些指标是否符合预期。
