# Global + Instance Scheduler 脚本说明

## 0. 先看这里：如果你现在就要运行

### 0.1 运行前你至少要检查这几个配置

默认 Qwen 基准配置文件是：

- `/home/tianzhu/vllm-omni/global_scheduler.qwen.yaml`

如果你要跑 Wan2.2，对应 base 配置文件是：

- `/home/tianzhu/vllm-omni/global_scheduler.wan2_2.yaml`

在真正运行前，至少确认下面这些字段已经改成你机器上真实可用的值：

- `benchmark.model`
  - 你要压测的模型名或本地模型路径
- `benchmark.dataset`
  - 当前 `global_scheduler.qwen.yaml` 默认是 `random`；如果你要跑 trace，需要改成 `trace`
- `benchmark.dataset_path`
  - 当 `dataset=trace` 时，这里是 trace 文件路径
- `benchmark.random_request_config`
  - 当 `dataset=random` 时，这里决定随机混合请求分布
- `benchmark.worker_ids`
  - 本次实验要启用哪些 worker
- `instances[].endpoint`
  - 每个 worker 实际监听的地址和端口
- `instances[].launch.model`
  - 每个 worker 启动时使用的模型
- `instances[].launch.env.CUDA_VISIBLE_DEVICES`
  - 每个 worker 实际绑定哪些 GPU
- `instances[].stop.args`
  - 停止命令是否和你的模型、端口匹配

如果你要跑当前默认目标组合：

- 全局调度：`min_queue_length`
- 实例内调度：`sjf`
- step chunk：开启
- chunk preemption：开启

那么 `global_scheduler.qwen.yaml` 本身已经会提供这一组默认模板；你最常需要改的是：

- 模型路径
- random 混合请求配置
- 如果你切回 `trace`，再改 trace 路径
- GPU 映射
- worker 数量和端口

### 0.2 跑单个默认组合

如果你只想跑当前默认组合，直接执行：

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml \
REQUEST_DURATION_S=600 \
REQUEST_RATES=0.2,0.4,0.6 \
benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh
```

这条命令会做这些事：

1. 从 `global_scheduler.qwen.yaml` 读入 base 配置
2. 生成临时 `global_scheduler.generated.yaml`
3. 用默认组合注入策略：
   - global `min_queue_length`
   - instance `sjf`
   - `step_chunk=1`
   - `chunk_preemption=1`
4. 启动 scheduler
5. 按多个 RPS 跑 benchmark
6. 输出结果目录

### 0.3 跑多个组合做对比

如果你想批量比较多个组合，执行：

```bash
CASE_MATRIX=$'qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4\nqwen_rr_sjf_preempt|round_robin|sjf|1|1|4' \
BASE_CONFIG=./global_scheduler.qwen.yaml \
REQUEST_DURATION_S=600 \
REQUEST_RATES=0.2,0.4,0.6 \
benchmarks/diffusion/scripts/run_global_instance_scheduler_rps_bench.sh
```

其中每一行 `CASE_MATRIX` 的格式是：

```text
case_name|global_policy|instance_policy|enable_step_chunk|enable_chunk_preemption|chunk_budget_steps
```

### 0.4 如果你需要改配置，优先改哪里

#### 只想改“压测流量”

不用改 YAML，优先改环境变量：

- `REQUEST_RATES`
- `REQUEST_DURATION_S`

当前默认值：

- `REQUEST_DURATION_S=600`

含义是：

- 每个 RPS 档默认跑 `600s`
- 当前联合实验脚本优先按“时长驱动”跑，不是固定每档请求数

#### 只想改“策略组合”

不用手改 YAML，优先改环境变量：

- `GLOBAL_POLICY`
- `INSTANCE_POLICY`
- `ENABLE_STEP_CHUNK`
- `ENABLE_CHUNK_PREEMPTION`
- `CHUNK_BUDGET_STEPS`
- `IMAGE_CHUNK_BUDGET_STEPS`
- `VIDEO_CHUNK_BUDGET_STEPS`
- `SMALL_REQUEST_LATENCY_THRESHOLD_MS`

或者在批量模式里改：

- `CASE_MATRIX`

#### 想改“模型 / worker / 数据集”

优先改 `global_scheduler.qwen.yaml` 或 `global_scheduler.wan2_2.yaml` 里的这些位置：

- `benchmark.model`
- `benchmark.backend`
- `benchmark.task`
- `benchmark.dataset`
- `benchmark.dataset_path`
- `benchmark.random_request_config`
- `benchmark.worker_ids`
- `instances[].endpoint`
- `instances[].backends`
- `instances[].launch.model`
- `instances[].launch.args`
- `instances[].launch.env`
- `instances[].stop.args`

#### 想切到 Wan2.2

至少要改：

- `benchmark.model`
- `benchmark.backend`
- `benchmark.task`
- `benchmark.dataset_path`
- `benchmark.random_request_config`
- `instances[].backends`
- `instances[].launch.model`
- `instances[].launch.args`
- `instances[].launch.env.CUDA_VISIBLE_DEVICES`

同时要把 worker 启动参数切成 Wan2.2 对应的并行配置。当前可以参考：

- `/home/tianzhu/vllm-omni/benchmarks/diffusion/scripts/run_wan_sp4_cfg2_hsdp_rps_bench.sh`

本文档说明 `benchmarks/diffusion/scripts/` 下与“全局调度 + 实例内调度”联合实验相关的脚本职责，以及它们之间的调用关系。

当前目标组合主要是：

- 全局调度：`min_queue_length`
- 实例内调度：`sjf`
- 实例内抢占：`diffusion_enable_step_chunk + diffusion_enable_chunk_preemption`

## 1. 相关脚本清单

本轮新增的脚本：

- `benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh`
- `benchmarks/diffusion/scripts/run_global_instance_scheduler_rps_bench.sh`

它们复用的已有脚本：

- `scripts/run_global_scheduler_benchmark_one_shell.sh`
- `scripts/run_global_scheduler_benchmark.sh`

它们最终调用的 benchmark 主程序：

- `benchmarks/diffusion/diffusion_benchmark_serving.py`

## 2. 每个脚本的职责

### 2.1 `run_global_instance_scheduler_rps_bench.sh`

这是“批量入口脚本”。

它负责：

- 定义要跑哪些 case
- 为每个 case 分配输出目录
- 调用单 case 脚本逐个执行
- 在所有 case 跑完后汇总 `summary.json` / `summary.csv`

你可以把它理解为：

- case 组织器
- suite 级目录管理器
- 最终结果汇总器

它不直接解析或改写 YAML，也不直接启动 scheduler。

### 2.2 `run_global_instance_scheduler_case.sh`

这是“单 case 执行器”。

它负责：

- 读取一个 base config
- 生成一个临时 `global_scheduler.generated.yaml`
- 在临时 YAML 里覆写：
  - `policy.baseline.algorithm`
  - 目标 worker 的 `launch.args`
- 把实例内策略改成指定组合，例如：
  - `sjf`
  - `sjf + step_chunk + preemption`
- 然后调用已有的 scheduler benchmark runner 去真正执行实验

你可以把它理解为：

- 单轮实验配置生成器
- 策略注入器
- 单轮实验触发器

### 2.3 `scripts/run_global_scheduler_benchmark_one_shell.sh`

这是“scheduler 生命周期包装器”。

它负责：

- 从 `CONFIG_FILE` 读取 scheduler 地址
- 启动 global scheduler 进程
- 等 `/health` ready
- 再调用下一级 benchmark 脚本
- benchmark 完成后清理 scheduler 进程

它不修改配置内容，只负责“把 scheduler 和 benchmark 放到一个 shell 生命周期里”。

### 2.4 `scripts/run_global_scheduler_benchmark.sh`

这是“多 RPS benchmark 执行器”。

它负责：

- 从 `CONFIG_FILE` 读取 `benchmark` 配置块
- 等 worker 变成 `routable=true` 且 `/v1/models` ready
- 解析 `REQUEST_RATES`
- 对每个 RPS 调用一次 `diffusion_benchmark_serving.py`

它不负责任何策略注入；它只消费最终 YAML。

### 2.5 `diffusion_benchmark_serving.py`

这是最底层的 benchmark 主程序。

它负责：

- 按指定数据集构造请求
- 以指定 `request-rate` 和 `max-concurrency` 发请求
- 收集吞吐、延迟和可选 SLO 指标
- 写出每轮 metrics JSON

## 3. 调用关系

当前完整调用链是：

```text
run_global_instance_scheduler_rps_bench.sh
  -> run_global_instance_scheduler_case.sh
    -> scripts/run_global_scheduler_benchmark_one_shell.sh
      -> scripts/run_global_scheduler_benchmark.sh
        -> benchmarks/diffusion/diffusion_benchmark_serving.py
```

如果展开成“每一层做什么”，可以写成：

```text
批量 case 管理
  -> 生成单 case 临时 YAML 并注入策略
    -> 启动/停止 scheduler
      -> 按多个 RPS 循环调用 benchmark
        -> 真正发请求并写 metrics
```

## 4. 为什么要拆成两层新脚本

没有把所有逻辑塞进一个脚本，主要是为了把“策略组合”和“批量实验控制”拆开。

这样拆的好处是：

- `run_global_instance_scheduler_case.sh`
  - 更适合调试单个组合
  - 可以快速验证某一个具体配置是否能跑通
- `run_global_instance_scheduler_rps_bench.sh`
  - 更适合批量跑多个组合
  - 不需要重复写 case 目录和汇总逻辑

也就是说：

- 想验证一个组合是否能工作，用 `case`
- 想系统比较多个组合，用 `rps_bench`

## 5. 当前新脚本实际会改哪些配置

`run_global_instance_scheduler_case.sh` 当前会在临时 YAML 中改这些字段：

- `policy.baseline.algorithm`
- `benchmark.output_file`
- 可选：
  - `benchmark.worker_ids`
  - `benchmark.model`
  - `benchmark.backend`
  - `benchmark.task`
  - `benchmark.dataset`
  - `benchmark.dataset_path`
  - `benchmark.random_request_config`
  - `benchmark.max_concurrency`
  - `benchmark.warmup_requests`
  - `benchmark.warmup_num_inference_steps`

同时，它会重写目标 worker 的 `launch.args` 中这些实例内调度相关参数：

- `--instance-scheduler-policy`
- `--diffusion-enable-step-chunk`
- `--diffusion-enable-chunk-preemption`
- `--diffusion-chunk-budget-steps`
- `--diffusion-image-chunk-budget-steps`
- `--diffusion-video-chunk-budget-steps`
- `--diffusion-small-request-latency-threshold-ms`

然后按当前 case 重新注入：

- `--instance-scheduler-policy <INSTANCE_POLICY>`
- `--diffusion-enable-step-chunk`（如果开启）
- `--diffusion-enable-chunk-preemption`（如果开启）
- `--diffusion-chunk-budget-steps <CHUNK_BUDGET_STEPS>`
- `--diffusion-image-chunk-budget-steps <IMAGE_CHUNK_BUDGET_STEPS>`（如果设置；否则继承 base YAML）
- `--diffusion-video-chunk-budget-steps <VIDEO_CHUNK_BUDGET_STEPS>`（如果设置；否则继承 base YAML）
- `--diffusion-small-request-latency-threshold-ms <SMALL_REQUEST_LATENCY_THRESHOLD_MS>`（如果设置；否则继承 base YAML）

这意味着：

- base YAML 可以作为模板保留
- 不同实验组合通过临时生成 YAML 来切换
- 不需要你手动反复改主配置文件

## 6. 当前推荐的使用方式

### 6.1 跑单个 case

适合调试：

```bash
GLOBAL_POLICY=min_queue_length \
INSTANCE_POLICY=sjf \
ENABLE_STEP_CHUNK=1 \
ENABLE_CHUNK_PREEMPTION=1 \
CHUNK_BUDGET_STEPS=4 \
SMALL_REQUEST_LATENCY_THRESHOLD_MS=1200 \
REQUEST_DURATION_S=600 \
REQUEST_RATES=0.2,0.4,0.6 \
BASE_CONFIG=./global_scheduler.qwen.yaml \
benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh
```

### 6.2 跑一整组 case

适合批量对比：

```bash
CASE_MATRIX=$'qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4\nqwen_rr_sjf_preempt|round_robin|sjf|1|1|4' \
REQUEST_DURATION_S=600 \
REQUEST_RATES=0.2,0.4,0.6 \
BASE_CONFIG=./global_scheduler.qwen.yaml \
benchmarks/diffusion/scripts/run_global_instance_scheduler_rps_bench.sh
```

## 7. 当前输出目录结构

如果跑批量脚本，结果通常会长这样：

```text
benchmarks/diffusion/results/<suite_name>/
  summary.json
  summary.csv
  <case_name_1>/
    global_scheduler.generated.yaml
    global_scheduler_server.log
    metrics_rps_0p2.json
    metrics_rps_0p4.json
  <case_name_2>/
    global_scheduler.generated.yaml
    global_scheduler_server.log
    metrics_rps_0p2.json
    metrics_rps_0p4.json
```

其中：

- 每个 case 目录保存该 case 生成出来的临时配置和运行结果
- suite 根目录保存所有 case 的汇总结果

## 8. 如何理解“新脚本”和旧脚本的边界

可以简单记成：

- 新脚本负责：
  - 组合实验
  - 策略注入
  - case/suite 组织
- 旧脚本负责：
  - scheduler 生命周期
  - worker ready 检查
  - 多 RPS benchmark 执行
- benchmark 主程序负责：
  - 发请求
  - 收指标

换句话说，本轮新增脚本不是替换旧脚本，而是在旧脚本之上加了一层“联合实验编排”。

## 9. 后续扩展到 Wan2.2 时，哪些地方不需要重写

切换到 Wan2.2 时，调用链本身不需要变，仍然是：

```text
run_global_instance_scheduler_rps_bench.sh
  -> run_global_instance_scheduler_case.sh
    -> run_global_scheduler_benchmark_one_shell.sh
      -> run_global_scheduler_benchmark.sh
        -> diffusion_benchmark_serving.py
```

需要替换的主要是传入参数和 base YAML：

- `benchmark.model`
- `benchmark.backend`
- `benchmark.task`
- `benchmark.dataset_path`
- `benchmark.random_request_config`
- `instances[].launch.model`
- `instances[].launch.args`
- `instances[].backends`

不需要重写脚本层次结构。
