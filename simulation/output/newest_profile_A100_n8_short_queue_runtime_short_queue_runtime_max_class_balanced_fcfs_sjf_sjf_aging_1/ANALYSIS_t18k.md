# t_end=18000s 仿真严格分析报告

## 1. 实验配置

- **t_end**: 18000s（5 小时）
- **rps**: [0.2, 0.4, 0.6, 0.8, 1.0]
- **全局调度**: short_queue_runtime / short_queue_runtime_max_class_balanced
- **实例调度**: fcfs, sjf, sjf_aging_0.15, sjf_aging_0.2, sjf_aging_0.25

## 2. 高负载 (rps=1.0) 全组合指标

| Algorithm_Policy | P95(s) | Mean(s) | QPS |
|------------------|--------|---------|-----|
| short_queue_runtime_fcfs | 9633.6 | 5051.4 | 0.6382 |
| short_queue_runtime_sjf | 12800.8 | 1734.7 | 0.6382 |
| short_queue_runtime_sjf_aging_0.15 | **9535.3** | 4959.5 | 0.6380 |
| short_queue_runtime_sjf_aging_0.2 | 9565.1 | 4984.6 | 0.6380 |
| short_queue_runtime_sjf_aging_0.25 | 9575.6 | 4999.0 | 0.6383 |
| short_queue_runtime_max_class_balanced_fcfs | 9632.2 | 5051.1 | 0.6381 |
| short_queue_runtime_max_class_balanced_sjf | 12784.9 | 1726.2 | 0.6379 |
| short_queue_runtime_max_class_balanced_sjf_aging_0.15 | **9529.4** | 4957.0 | 0.6382 |
| short_queue_runtime_max_class_balanced_sjf_aging_0.2 | 9558.3 | 4981.9 | 0.6381 |
| short_queue_runtime_max_class_balanced_sjf_aging_0.25 | 9574.7 | 4996.4 | 0.6381 |

## 3. sjf_aging_factors 最优选择

| 全局算法 | 最优 policy | P95(s) |
|----------|-------------|--------|
| short_queue_runtime | sjf_aging_0.15 | 9535.3 |
| short_queue_runtime_max_class_balanced | sjf_aging_0.15 | **9529.4** |

**结论：sjf_aging_factor=0.15 为 P95 最优，优于 0.2 和 0.25。**

## 4. max_class_balanced 可行性（rps=1.0 配对对比）

| Policy | Base_P95 | Bal_P95 | P95_Delta |  verdict |
|--------|----------|---------|-----------|----------|
| fcfs | 9633.6 | 9632.2 | -0.0% | 可行 |
| sjf | 12800.8 | 12784.9 | -0.1% | 可行 |
| sjf_aging_0.15 | 9535.3 | 9529.4 | -0.1% | 可行 |
| sjf_aging_0.2 | 9565.1 | 9558.3 | -0.1% | 可行 |
| sjf_aging_0.25 | 9575.6 | 9574.7 | -0.0% | 可行 |

**5/5 policies 下 max_class_balanced 均不劣于 base，全部可行。**

## 5. 全 RPS max_class_balanced 收益汇总

| Policy | 胜出次数 | 平均 P95 变化 |
|--------|----------|---------------|
| fcfs | 5/5 | **-3.4%** |
| sjf | 4/5 | **-2.7%** |
| sjf_aging_0.15 | 5/5 | **-2.7%** |
| sjf_aging_0.2 | 5/5 | **-2.7%** |
| sjf_aging_0.25 | 5/5 | **-2.5%** |

低负载 (rps 0.2–0.6) 下 P95 改善更显著；高负载下轻微改善或持平。

## 6. 全 RPS P95 详细（fcfs / sjf / sjf_aging_0.15）

| RPS | base_fcfs | bal_fcfs | base_sjf | bal_sjf | base_aging | bal_aging |
|-----|-----------|----------|----------|---------|------------|-----------|
| 0.20 | 46.3 | 44.8 | 46.3 | 44.8 | 46.3 | 44.8 |
| 0.40 | 48.1 | 47.2 | 48.1 | 47.2 | 48.1 | 47.2 |
| 0.60 | 64.4 | 56.5 | 60.4 | 55.1 | 59.9 | 54.8 |
| 0.80 | 4282.2 | 4281.9 | 6162.5 | 6174.2 | 4214.9 | 4211.2 |
| 1.00 | 9633.6 | 9632.2 | 12800.8 | 12784.9 | 9535.3 | 9529.4 |

## 7. 最终推荐

1. **sjf_aging_factor**: 推荐 **0.15**（P95 最优）
2. **max_class_balanced**: **推荐默认启用**
   - 5/5 policies 下均有收益或持平
   - 全 RPS 平均 P95 改善 2.5%–3.4%
   - 与 t_end=1800s 短跑结论不同：长跑 (18000s) 下 max_class_balanced 收益稳定
3. **最佳组合**: `short_queue_runtime_max_class_balanced` + `sjf_aging_0.15`
   - 高负载 P95 = 9529.4s（当前配置下最优）
