# Real Execution Benchmark Results

本目录包含在 **单实例 A100 GPU** 上对 Qwen-Image 文生图模型进行的真实 benchmark 结果，
用于与仿真模拟器 (`simulation.py`) 的输出做对齐验证。

## 测试环境

| 项目 | 值 |
|---|---|
| 模型 | Qwen-Image (文生图 diffusion) |
| GPU | NVIDIA A100 |
| 实例数 | 1 (worker0) |
| 后端 | vllm-omni |
| 全局调度 | round_robin (单实例，全局调度无影响) |
| 最大并发 | 32 (`--diffusion-engine-max-concurrency 32`) |
| VAE 优化 | `--vae-use-slicing --vae-use-tiling` |
| Warmup | 每轮 rps 测试前 8 个 warmup 请求 (1 step) |
| 每轮请求数 | 100 |
| 数据集 | random (按权重随机采样四类请求) |

## 请求类型分布

benchmark 使用 `--dataset random` 模式，按以下权重随机生成四类请求：

| 尺寸 | Steps | 权重 | A100 Service Time |
|---|---|---|---|
| 512x512 | 20 | 0.15 | ~2.50s |
| 768x768 | 20 | 0.25 | ~4.79s |
| 1024x1024 | 25 | 0.45 | ~11.07s |
| 1536x1536 | 35 | 0.15 | ~35.91s |

> A100 hardware profile 见 `simulation/profile/qwen_A100.json`。

## 测试的算法

共测试 **9 种实例内调度策略**，分为三类：

### 非抢占算法

| 算法 | `--instance-scheduler-policy` | Step Chunk | Preemption | chunk_budget_steps | small_request_threshold_ms |
|---|---|---|---|---|---|
| **fcfs** | `fcfs` | 否 | 否 | 12 | 12000 |
| **sjf_aging_no_preempt** | `sjf_aging` | 否 | 否 | 12 | 12000 |
| **size_bucket_sjf_aging_no_preempt** | `size_bucket_sjf_aging` | 否 | 否 | 12 | 12000 |

### 抢占算法 (chunk_budget_steps=12)

| 算法 | `--instance-scheduler-policy` | Step Chunk | Preemption | chunk_budget_steps | small_request_threshold_ms |
|---|---|---|---|---|---|
| **sjf_aging** | `sjf_aging` | 是 | 是 | 12 | 12000 |
| **size_bucket_sjf_aging** | `size_bucket_sjf_aging` | 是 | 是 | 12 | 12000 |
| **p95_first** | `p95-first` | 是 | 是 | 12 | 12000 |
| **p95_bucket_sjf** | `p95-bucket-sjf` | 是 | 是 | 12 | 12000 |
| **p95_bucket_sjf_norm** | `p95-bucket-sjf-normalized` | 是 | 是 | 12 | 12000 |

### 特殊配置

| 算法 | `--instance-scheduler-policy` | Step Chunk | Preemption | chunk_budget_steps | small_request_threshold_ms | 备注 |
|---|---|---|---|---|---|---|
| **sjf_aging_guarded** | `sjf_aging_guarded` | 是 | 是 | **5** | **6000** | chunk budget 更小，小请求阈值更低；仅跑了 rps=0.125 |

## RPS 测试矩阵

每个算法在不同 rps 下各发送 100 个请求。各算法覆盖的 rps 级别如下：

| 算法 | rps=0.05 | rps=0.075 | rps=0.125 |
|---|---|---|---|
| fcfs | 100 reqs | 100 reqs | 100 reqs |
| p95_bucket_sjf | 100 reqs | 100 reqs | 100 reqs |
| p95_bucket_sjf_norm | 100 reqs | 100 reqs | 100 reqs |
| p95_first | 100 reqs | 100 reqs | 100 reqs |
| size_bucket_sjf_aging | 100 reqs | 100 reqs | 100 reqs |
| size_bucket_sjf_aging_no_preempt | 100 reqs | - | - |
| sjf_aging | 100 reqs | 100 reqs | 100 reqs |
| sjf_aging_guarded | - | - | 100 reqs |
| sjf_aging_no_preempt | 100 reqs | 100 reqs | 100 reqs |

> `size_bucket_sjf_aging_no_preempt` 仅完成 rps=0.05 的测试。
> `sjf_aging_guarded` 仅完成 rps=0.125 的测试。

## 性能汇总

### rps=0.05 (低负载)

| 算法 | Mean Latency | P95 Latency | Mean Wait | Mean Service | Throughput |
|---|---|---|---|---|---|
| fcfs | 16.642s | 37.337s | 5.390s | 11.252s | 0.0490 |
| p95_bucket_sjf | 13.961s | 35.852s | 2.692s | 11.269s | 0.0490 |
| p95_bucket_sjf_norm | 13.650s | 40.633s | 2.150s | 11.501s | 0.0490 |
| p95_first | 13.931s | 35.940s | 2.588s | 11.344s | 0.0490 |
| size_bucket_sjf_aging | 13.931s | 35.942s | 2.585s | 11.346s | 0.0490 |
| size_bucket_sjf_aging_no_preempt | 14.221s | 35.921s | 2.970s | 11.251s | 0.0490 |
| sjf_aging | 14.019s | 35.921s | 2.720s | 11.300s | 0.0490 |
| sjf_aging_no_preempt | 14.181s | 35.864s | 2.949s | 11.232s | 0.0490 |

### rps=0.075 (中负载)

| 算法 | Mean Latency | P95 Latency | Mean Wait | Mean Service | Throughput |
|---|---|---|---|---|---|
| fcfs | 22.794s | 56.196s | 11.541s | 11.254s | 0.0715 |
| p95_bucket_sjf | 21.052s | 56.446s | 8.975s | 12.077s | 0.0715 |
| p95_bucket_sjf_norm | 18.258s | 61.929s | 6.216s | 12.042s | 0.0715 |
| p95_first | 20.389s | 56.158s | 8.891s | 11.498s | 0.0715 |
| size_bucket_sjf_aging | 21.194s | 57.035s | 9.706s | 11.488s | 0.0715 |
| sjf_aging | 21.028s | 56.044s | 9.325s | 11.703s | 0.0715 |
| sjf_aging_no_preempt | 22.209s | 56.911s | 10.978s | 11.231s | 0.0715 |

### rps=0.125 (高负载)

| 算法 | Mean Latency | P95 Latency | Mean Wait | Mean Service | Throughput |
|---|---|---|---|---|---|
| fcfs | 126.513s | 268.700s | 115.252s | 11.261s | 0.0881 |
| p95_bucket_sjf | 82.733s | 312.058s | 68.759s | 13.973s | 0.0884 |
| p95_bucket_sjf_norm | 60.450s | 338.816s | 46.610s | 13.839s | 0.0881 |
| p95_first | 77.803s | 313.445s | 66.503s | 11.300s | 0.0883 |
| size_bucket_sjf_aging | 79.733s | 320.052s | 68.314s | 11.419s | 0.0877 |
| sjf_aging | 78.505s | 312.524s | 67.162s | 11.343s | 0.0883 |
| sjf_aging_guarded | 122.856s | 267.143s | 110.834s | 12.021s | 0.0882 |
| sjf_aging_no_preempt | 80.836s | 296.705s | 69.610s | 11.226s | 0.0884 |

## 四类请求 Hardware Profile (fcfs 基准)

在无抢占的 fcfs 算法下测得的纯硬件执行时间：

| 尺寸 | Steps | Mean Service Time | Min | Max | 样本数 |
|---|---|---|---|---|---|
| 512x512 | 20 | 2.504s | 2.484s | 2.664s | 45 |
| 768x768 | 20 | 4.793s | 4.767s | 5.159s | 96 |
| 1024x1024 | 25 | 11.072s | 11.031s | 11.699s | 117 |
| 1536x1536 | 35 | 35.912s | 35.831s | 35.991s | 42 |

## 目录结构

```
simulation/real_execution/
├── README.md                        # 本文件
├── <algorithm>/
│   ├── <algorithm>_rps_<rps>_requests.csv   # 每个 rps 的逐请求明细 (按完成时间排序)
│   ├── <algorithm>_rps_<rps>.trace.txt      # 请求序列 (按到达时间排序，供仿真输入)
│   ├── <algorithm>_rps_<rps>.json           # 每个 rps 的统计摘要
│   └── profile.json                         # 四类请求的平均 service_time (结构同 qwen_A100.json)
```

### CSV 字段说明

| 字段 | 含义 |
|---|---|
| algorithm | 算法名 |
| rps | 请求到达速率 |
| request_id | 按到达顺序分配的数字 ID (0-99)，与 trace 文件中的 request_id 一一对应 |
| arrival_time | 到达时间 (相对该 rps 组首个请求, 秒) |
| start_time | 开始执行时间 (相对, 秒) |
| finish_time | 完成时间 (相对, 秒) |
| latency | 端到端延迟 = finish - arrival (秒) |
| waiting_time | 等待时间 = start - arrival (秒) |
| service_time | 服务时间 = finish - start (挂钟时间；抢占算法下包含被抢占后空等的时间，非纯 GPU 计算时间) |
| size | 分辨率 (如 1024x1024) |
| steps | 推理步数 |
| executed_steps | 实际执行步数 (抢占后续做，不重头开始，故 executed_steps == total_steps) |
| dispatch_epoch | 被调度的轮次 (抢占后 +1) |

### JSON 字段说明

与仿真输出格式一致，包含 `latency_mean/median/p50/p95/p99`、`waiting_time_mean/p95/p99`、`service_time_mean/p95/p99`、`throughput_qps`、`duration` 等。

## 统计指标说明

### 数据来源与口径

本目录的 JSON/CSV 统计指标基于 **服务端** `worker0.log` 中的 `REQUEST_COMPLETED` 事件计算，
而原始 `metrics_rps_*.json` 由 **benchmark 客户端**测量。两者存在系统性偏移：

| 维度 | 服务端 (本目录) | 客户端 (原始 metrics) | 差异 |
|---|---|---|---|
| 单请求 latency | `completion_ts - arrival_ts` (scheduler 层) | 发送请求 → 收到响应 (含网络 RTT) | 客户端高 ~0.7s |
| duration | `last_completion - first_arrival` (服务端) | 发送首请求 → 收到末响应 | 客户端高 ~1.4s |
| throughput_qps | `completed / duration` | 同 | 基本一致 |
| latency 分位数 | 线性插值 (与 numpy 默认一致) | 同 | 客户端高 ~0.7s |

偏移量高度一致：每请求 +0.71s (±0.05s)，完全由 HTTP 网络 RTT 解释。

### 统计指标计算口径

```
duration         = max(completion_ts) - min(arrival_ts)    # 首请求到达 → 末请求完成
throughput_qps   = completed_requests / duration
latency          = completion_ts - arrival_ts              # 每请求端到端延迟
latency_mean     = mean(latencies)
latency_p50/p95/p99 = percentile(latencies, p)            # 线性插值
```

### waiting_time 与 service_time 的口径差异 (仿真 vs 实测)

在有抢占的算法下，实测与仿真的 `waiting_time` / `service_time` 语义不同：

| 指标 | 实测 | 仿真 |
|---|---|---|
| service_time | `completion_ts - first_dispatch_ts` (挂钟时间，含抢占等待) | `sum(chunk_service_times)` (纯 GPU 计算时间) |
| waiting_time | `first_dispatch_ts - arrival_ts` (仅首次派发前等待) | `latency - service_time` (所有非计算时间) |

对于 **fcfs (无抢占)** 场景，两者等价。`latency` 及其所有分位数口径完全一致，可直接对比。

## 仿真对齐要点

在仿真器中复现这些结果时，需注意：

1. **Profile**: 使用 `simulation/profile/qwen_A100.json` 作为硬件执行时间参数
2. **实例数**: 1 (与真实测试一致)
3. **数据集**: 每个 rps 使用对应的 `.trace.txt` 文件 (按 arrival_time 升序排列的请求序列)
4. **全局调度**: `round_robin` (单实例下无差异)
5. **RPS**: `[0.05, 0.075, 0.125]`
6. **每轮请求数**: 100
7. **抢占参数**: `chunk_budget_steps=12`, `small_request_threshold_ms=12000` (sjf_aging_guarded 除外: budget=5, threshold=6000)

## 数据来源

原始日志位于 `C:\Users\Asus\Desktop\real_execution\<algorithm>\instance_logs\worker0.log`，
由 `simulation/tools/extract_real_execution.py` 脚本解析生成。
统计指标全部从提取的服务端数据独立计算，未复制客户端 metrics 文件。
