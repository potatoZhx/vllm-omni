# Global Scheduler + Instance Scheduler 联合实验说明

本文档说明如何在当前仓库里运行一套联合调度实验：

- 全局调度：`min_queue_length`
- 实例内调度：`sjf`
- 实例内抢占：`diffusion_enable_step_chunk + diffusion_enable_chunk_preemption`

当前主目标是 `Qwen/Qwen-Image`。文档最后会补充后续切换到 `Wan2.2` 时需要替换的配置位。

本文档关注的是“当前仓库里需要配哪些项、这些项分别在哪里生效、推荐如何组织运行流程”，不是单独介绍 benchmark 脚本实现细节。

## 1. 实验目标与链路

这套实验的完整链路是：

1. benchmark client 按指定 RPS 发送 diffusion 请求
2. global scheduler 根据全局策略在多个 worker 间路由
3. 被选中的 worker 再用实例内调度策略排序和执行本地等待队列
4. worker 在 chunk 边界可重新入队，从而让 `sjf + preemption` 真正生效

对于本文档的目标组合，实际决策分成两层：

- 第 1 层，全局层：
  - `policy.baseline.algorithm=min_queue_length`
  - 在可路由实例中选择当前 `queue_len` 最小的实例
- 第 2 层，实例层：
  - `--instance-scheduler-policy sjf`
  - 在单个实例的等待队列里按剩余估计耗时排序
  - 配合 step chunk 和 chunk preemption，让长请求能够在 chunk 边界重新排队

## 2. 需要配置的三类对象

要跑通这套实验，至少要同时配置三类对象：

1. global scheduler 服务配置
2. worker 实例启动配置
3. benchmark 运行配置

三者的职责不要混淆：

- global scheduler 配置决定：
  - 有哪些 worker
  - 全局选路算法是什么
  - scheduler 如何拉起 / 停止 worker
  - benchmark 默认读取哪些模型、数据集、worker id
- worker 启动配置决定：
  - 模型是什么
  - 实例内调度算法是什么
  - 是否开启 step chunk / preemption
  - 单实例的并行和分布式参数是什么
- benchmark 配置决定：
  - 发多少请求
  - 请求来自哪个数据集
  - 以什么 RPS 压测
  - 结果写到哪里

## 3. Qwen-image 当前所需最小配置

### 3.1 global scheduler 层

Qwen 对应文件通常是：

- `global_scheduler.qwen.yaml`

你至少需要这些顶层块：

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 600
  instance_health_check_interval_s: 5.0
  instance_health_check_timeout_s: 1.0

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: min_queue_length

benchmark:
  worker_ids: [worker0, worker1, worker2, worker3, worker4, worker5, worker6, worker7]
  worker_ready_timeout_s: 1200
  model: Qwen/Qwen-Image
  backend: vllm-omni
  task: t2i
  dataset: random
  random_request_config: >-
    [
      {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
      {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
      {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
      {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
    ]
  # dataset_path: ./benchmarks/dataset/sd3_trace_redistributed.txt
  max_concurrency: 20
  warmup_requests: 1
  warmup_num_inference_steps: 1
  output_file: ./logs/global_scheduler_qwen_image_benchmark_metrics.json
  auto_stop: true
```

这些字段的作用分别是：

- `policy.baseline.algorithm`
  - 这里必须设成 `min_queue_length`
  - 这是全局调度算法，不影响实例内部的队列策略
- `benchmark.worker_ids`
  - 指定本次实验使用哪些 worker
- `benchmark.backend`
  - 对 Qwen-image，推荐写成 `vllm-omni`
  - 这样可以和你后续实际传给 benchmark 的 `--backend vllm-omni` 保持一致
- `benchmark.task`
  - 对 Qwen-image 文生图，通常是 `t2i`
- `benchmark.dataset`
  - 当前默认是 `random`
- `benchmark.dataset_path`
  - 当 `dataset=trace` 时，指向具体 trace 文件
- `benchmark.random_request_config`
  - 当 `dataset=random` 时，定义混合请求分布
- `benchmark.max_concurrency`
  - 这是 client 侧最大并发，不是 worker 内部并发
- `benchmark.auto_stop`
  - 当前可以先视为 benchmark 配置块里的保留字段，不是这轮联合实验的核心控制项

### 3.2 worker 层

每个实例都需要在 `instances:` 下配置一个条目。对 Qwen-image，推荐至少包含：

```yaml
instances:
  - id: worker0
    endpoint: http://127.0.0.1:8001
    instance_type: qwen-image
    numa_node: 0
    backends: [vllm-omni, openai]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args:
        - --omni
        - --max-concurrency
        - "2"
        - --instance-scheduler-policy
        - sjf
        - --diffusion-enable-step-chunk
        - --diffusion-enable-chunk-preemption
        - --diffusion-chunk-budget-steps
        - "4"
        - --ulysses-degree
        - "1"
        - --cfg-parallel-size
        - "1"
        - --vae-use-slicing
        - --vae-use-tiling
        - --num-weight-load-threads
        - "4"
      env:
        CUDA_VISIBLE_DEVICES: "0"
    stop:
      executable: pkill
      args:
        - -f
        - "vllm serve Qwen/Qwen-Image --port {endpoint_port}"
```

当前仓库默认模板是 8 个单卡 worker：`worker0` 到 `worker7`。

如果你要继续扩容或按机器实际拓扑调整，主要替换：

- `id`
- `endpoint`
- `numa_node`
- `CUDA_VISIBLE_DEVICES`

#### 这些 worker 参数里，最关键的是哪几个

- `--instance-scheduler-policy sjf`
  - 实例内等待队列按剩余估计耗时排序
- `--diffusion-enable-step-chunk`
  - 开启 step chunk 执行
- `--diffusion-enable-chunk-preemption`
  - 允许 unfinished request 在 chunk 边界重新入队
- `--diffusion-chunk-budget-steps 4`
  - 每次最多执行多少 step 再回到调度器
- `--diffusion-image-chunk-budget-steps`
  - 如果设置，会覆盖图像请求使用的 chunk budget
- `--diffusion-video-chunk-budget-steps`
  - 如果设置，会覆盖视频请求使用的 chunk budget
- `--diffusion-small-request-latency-threshold-ms`
  - 估计剩余时延低于该阈值的请求会直接跑完，不再继续分 chunk

如果不同时打开：

- `--diffusion-enable-step-chunk`
- `--diffusion-enable-chunk-preemption`

那么 `sjf` 只能在“新请求进入等待队列”时重排，已经开始跑的长请求不会在执行中途让出机会，实验效果会明显弱很多。

#### `--max-concurrency` 在这里的意义

在当前 global scheduler 实现里，`launch.args` 里的 `--max-concurrency` 有两个作用：

1. 供 scheduler 估算该实例可容纳多少 inflight
2. 作为实验配置记录的一部分保留在 YAML

当前 scheduler 在真正拉起 `vllm serve` 子进程时，会把 `--max-concurrency` 从启动命令里剥掉，不直接透传给子进程。因此：

- 这个参数目前主要服务于 global scheduler 的运行时状态估计
- 不是 worker 内核调度逻辑的开关

### 3.3 benchmark 层

如果你直接调用 benchmark 主脚本，Qwen-image 联合实验的最小命令可以写成：

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8089 \
  --backend vllm-omni \
  --model Qwen/Qwen-Image \
  --task t2i \
  --dataset random \
  --random-request-config '[
    {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
    {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
    {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
    {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
  ]' \
  --num-prompts 100 \
  --request-rate 0.5 \
  --max-concurrency 20 \
  --warmup-requests 1 \
  --warmup-num-inference-steps 1 \
  --output-file ./logs/qwen_global_minqlen_instance_sjf_rps_0p5.json
```

这里的关键点是：

- `--base-url` 指向 global scheduler，不是单个 worker
- `--backend vllm-omni`
  - 对应 scheduler 的 `/v1/chat/completions`
- `--request-rate`
  - 控制目标 RPS
- `--max-concurrency`
  - 控制 client 侧最大并发
- `--dataset random`
  - 默认通过 `random_request_config` 生成混合 heterogeneous workload

如果你使用当前联合实验脚本而不是手动调用 benchmark，需要额外记住：

- 运行时长由环境变量 `REQUEST_DURATION_S` 控制
- 当前默认值是 `600`
- 含义是每个 RPS 档默认跑 `600s`
- 当前脚本优先按“时长驱动”计算请求数，而不是固定每档 `num_prompts`

## 4. 推荐的 Qwen-image 运行步骤

建议按这个顺序跑：

1. 先确认 `global_scheduler.qwen.yaml` 中：
   - 全局算法是 `min_queue_length`
   - worker 的 `launch.args` 里已经是 `sjf + step_chunk + preemption`
2. 启动 global scheduler：

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.qwen.yaml
```

3. 查看实例状态：

```bash
curl -sS http://127.0.0.1:8089/instances
```

4. 如果 worker 刚 auto-start，还没 ready，主动探针一次：

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

5. 确认至少一个实例已经：
   - `enabled=true`
   - `healthy=true`
   - `draining=false`
   - `process_state=running`
   - `routable=true`
6. 再开始跑 benchmark
7. 按不同 RPS 重复实验，比较每轮 metrics 输出

如果你直接用联合实验脚本，推荐像这样显式写出测试时长：

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml \
REQUEST_DURATION_S=600 \
REQUEST_RATES=0.2,0.4,0.6 \
benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh
```

## 5. 哪些参数属于“实验强相关”

对于你现在要做的联合实验，最重要的不是把所有参数都写满，而是优先锁定这些参数：

- 全局层：
  - `policy.baseline.algorithm=min_queue_length`
- 实例层：
  - `--instance-scheduler-policy sjf`
  - `--diffusion-enable-step-chunk`
  - `--diffusion-enable-chunk-preemption`
  - `--diffusion-chunk-budget-steps`
- benchmark 层：
  - `--request-rate`
  - `--max-concurrency`
  - `--dataset`
  - `--dataset-path` / `--random-request-config`

如果你后面要做横向对比，通常优先固定下面这些不变：

- 模型
- 数据集
- `random_request_config` 或 trace 文件
- chunk budget
- worker 数量
- 每个 worker 的 GPU 并行配置

然后只改变：

- 全局调度算法
- 实例内调度算法
- 是否开启 preemption
- RPS

## 6. 关于 random 与 trace 的建议

对当前仓库里的 Qwen 联合实验模板，默认推荐先用 `random`，原因是：

- 更容易直接复现实验，不依赖额外 trace 文件
- 可以通过 `random_request_config` 明确固定混合负载分布
- 对 `min_queue_length` 和 `sjf + preemption` 一样能制造异构请求

如果你切到 `trace` 模式，需要记住：

- `dataset` 改成 `trace`
- 再提供 `benchmark.dataset_path`
- `num_inference_steps` 等字段可以来自 trace
- 如果 CLI 显式传了 `--width` / `--height`，会覆盖 trace 里的对应值

因此，做联合实验时推荐：

- 默认先固定一份 `random_request_config`
- 如果要复现线上或历史负载，再切到 `trace`

## 7. 当前组合实验不需要的配置

以下配置在“global min_queue_length + instance sjf + preemption”这个目标里不是必需项：

- `policy.baseline.runtime_profile_path`
  - `min_queue_length` 不依赖 runtime profile
- `--instance-runtime-profile-path`
  - `sjf` 可以先只依赖启发式估时
- `--instance-runtime-profile-name`
  - 同上
- `--instance-scheduler-slo-target-ms`
  - 这是 deadline-aware 策略才需要重点关心的

后续如果你把实例内策略从 `sjf` 切到 `slo_first` / `slack_age` / `slack_cost_age`，这些配置才会变成一线配置。

## 8. 后续切换到 Wan2.2 时，需要替换哪些位置

后续切到 Wan2.2，不要重写整套逻辑，只替换同一组配置位。

### 8.1 需要替换的模型与 backend

- `benchmark.model`
- `instances[].launch.model`
- `benchmark.backend`
- `benchmark.task`
- `instances[].backends`

通常会变成：

- 模型：`Wan2.2` 对应的本地路径
- backend：`v1/videos`
- task：`t2v` / `i2v`
- backends：`[v1/videos]`

### 8.2 需要替换的并行启动参数

Wan2.2 一般还要替换：

- `--tensor-parallel-size`
- `--usp`
- `--ring`
- `--cfg-parallel-size`
- `--use-hsdp`
- `--hsdp-replicate-size`
- `--num-weight-load-threads`
- `CUDA_VISIBLE_DEVICES`

这些可以参考：

- `benchmarks/diffusion/scripts/run_wan_sp4_cfg2_hsdp_rps_bench.sh`

### 8.3 需要替换的数据集配置

Wan2.2 通常需要替换：

- `benchmark.dataset_path`
- `benchmark.random_request_config`
- `benchmark.task`
- trace 文件内容本身或 random 配置本身

因为视频请求和图像请求的：

- `num_frames`
- `fps`
- `num_inference_steps`

分布完全不同，不能直接共用 Qwen-image 的 trace。

### 8.4 运行逻辑本身不需要改

即使切换到 Wan2.2，整体运行逻辑仍然应该保持：

1. 配置 global scheduler
2. 在 worker 的 `launch.args` 里设置实例内调度策略
3. 启动 scheduler
4. 等 worker ready
5. 用 benchmark 按多 RPS 跑
6. 汇总每个 RPS 的指标

也就是说，后续变化主要是“模型和并行参数”，不是“实验流程”。

### 8.4 Wan T2V step-chunk 恢复状态修复说明

在 `Wan2.2 T2V` 上开启 `step_chunk + preemption` 时，`FlowUniPCMultistepScheduler` 会维护多步历史状态，例如 `model_outputs`、`timestep_list`、`last_sample`、`lower_order_nums`、`_step_index` 和 `this_order`。如果这些状态不随请求一起保存和恢复，不同分辨率请求在同一实例内交错执行后，可能在 resume 时复用到别的请求留下的 scheduler 历史，从而在 `multistep_uni_c_bh_update()` 中触发形状不一致错误，例如 `160 vs 106`。

当前修复已经在 `pipeline_wan2_2.py` 中补齐了 scheduler state capture/restore，Wan T2V 的 chunked request 会在每次 chunk 结束后保存完整 scheduler 状态，并在 resume 前恢复。

建议用混合分辨率请求做回归验证：

```bash
WORKER_IDS=worker0 \
BASE_CONFIG=./global_scheduler.wan2_2.yaml \
REQUEST_DURATION_S=30 \
REQUEST_RATES=0.2 \
ENABLE_STEP_CHUNK=1 \
ENABLE_CHUNK_PREEMPTION=1 \
BENCHMARK_RANDOM_REQUEST_CONFIG='[
  {"width":854,"height":480,"num_inference_steps":6,"num_frames":17,"fps":16,"weight":0.5},
  {"width":1280,"height":720,"num_inference_steps":6,"num_frames":17,"fps":16,"weight":0.5}
]' \
benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh
```

期望现象：

- 日志中仍然出现 `REQUEST_PREEMPTED` 和 `REQUEST_RESUMED`
- 不再出现 `The size of tensor a (...) must match the size of tensor b (...)`
- 请求成功完成，或只因与本问题无关的原因失败

## 9. 推荐输出目录命名

为了后续对比不同组合，建议结果目录名直接编码策略组合，例如：

```text
logs/
  qwen_minqlen_global_sjf_preempt/
    rps_0p2.json
    rps_0p4.json
    rps_0p6.json
```

或者在文件名里显式写：

```text
qwen_global_minqlen_instance_sjf_preempt_rps_0p4.json
```

这样后面再加入：

- `round_robin + fcfs`
- `min_queue_length + sjf`
- `min_queue_length + sjf + preempt`
- `short_queue_runtime + sjf + preempt`

时，不会把结果混在一起。

## 10. 当前文档对应的推荐实验基线

如果你要先完成第一轮联合实验，推荐先只固定这一组基线：

- 模型：`Qwen/Qwen-Image`
- global scheduler：`min_queue_length`
- instance scheduler：`sjf`
- step chunk：开启
- chunk preemption：开启
- chunk budget：`4`
- backend：`vllm-omni`
- task：`t2i`
- dataset：`random`

先把这组基线跑通，再去扩展：

- 不同 RPS
- 不同 worker 数量
- 不同 chunk budget
- Wan2.2 视频模型

这样实验面会更干净，也更容易定位瓶颈来自哪一层。
