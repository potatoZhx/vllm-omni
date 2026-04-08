# vLLM-Omni Diffusion Global Scheduler 中文总览

本文档面向当前这轮 `v18-base` 迁移中的 diffusion 多实例在线服务能力，给出一个统一的中文入口。
重点回答三个问题：

- 我们这次实现了什么
- 最小可运行配置是什么
- 具体怎么配置、怎么 benchmark、怎么排查问题分别该看哪份 README

如果你只想先跑通一个最小闭环，先看本文的“最小启动配置”部分；如果你需要理解实现细节或调整实验参数，再按文末的“文档导航”继续阅读。

## 项目概览

这轮工作在 `vLLM-Omni` 中补齐了一个面向 diffusion 图像 / 视频在线服务的最小 global scheduler 闭环，目标是：

- 支持多实例请求路由
- 保持实现足够简单，便于验证和 benchmark
- 与现有 worker 内部调度能力兼容
- 让 benchmark、orchestration、runtime server 共用一套配置入口

当前 global scheduler 的定位是 `pure routing`，不是一个带等待队列的全局排队器。请求进入 scheduler 后，会立刻选择一个可路由 worker 并转发；worker 内部如何排队、如何做 step-level 执行，仍由 worker 自己负责。

## 我们实现了什么

当前已经具备的核心能力如下。

### 1. Global Scheduler Runtime

- 提供独立的 global scheduler server，用于统一接收和转发 diffusion 请求
- 支持 3 种全局路由策略：
  - `min_queue_length`
  - `round_robin`
  - `short_queue_runtime`
- 支持 3 类 backend：
  - `vllm-omni`
  - `openai`
  - `v1/videos`
- 支持实例级 runtime bookkeeping，包括：
  - `inflight`
  - `outstanding_runtime_s`
  - `ewma_service_time_s`
- 支持实例生命周期与健康管理：
  - `enable`
  - `disable`
  - `start`
  - `stop`
  - `restart`
  - `reload`

### 2. Scheduler-Visible 请求元数据

当前请求可向 scheduler 暴露以下信息，用于路由或 runtime 估算：

- `width`
- `height`
- `num_frames`
- `num_inference_steps`
- `estimated_cost_s`
- `slo_ms`

其中，`estimated_cost_s` 是当前最重要的代价信号。若使用 `short_queue_runtime`，建议 benchmark 或调用方显式传入该字段，以便真正体现 runtime-aware 的选路行为。

### 3. Benchmark 能力

我们保留并整理了 diffusion 在线服务 benchmark 主入口：

- `benchmarks/diffusion/diffusion_benchmark_serving.py`

它支持：

- 图像 / 视频在线服务压测
- 吞吐、时延分位数统计
- 可选的 SLO 达成率评估
- `random`、`trace`、`vbench` 三类数据集模式
- warmup 推断与 `estimated_cost_s` 注入

### 4. Orchestrator 闭环

我们还提供了一个面向实验和回归验证的 orchestrator，用于自动串起：

1. 读取基础 YAML
2. 启动 global scheduler
3. 启动并探测 worker
4. 调用 benchmark 对 scheduler 发压
5. 汇总输出产物

它支持：

- 单 case 运行
- suite 批量运行
- 使用环境变量覆盖基础配置
- 为 worker 统一注入 diffusion scheduler backend
- 复用 benchmark warmup profile

## 当前边界与设计取舍

为了先得到一个可运行、可验证、可 benchmark 的最小闭环，这轮实现有明确的边界。

当前支持：

- 全局纯路由，不做 scheduler 侧等待
- 最小实例级 runtime 状态维护
- 基于 inflight 或 runtime 估算的轻量策略
- 与 worker 侧已有实例内策略兼容

当前不做：

- scheduler 侧等待 / admission blocking
- 全局容量 gating
- `fcfs`、`estimated_completion_time` 等额外全局策略
- 全局侧复杂实例内策略编排
- 对 `chunk_preemption`、`chunk_budget` 的 orchestration 依赖

如果你在阅读配置时看到 `queue_len` 相关语义，需要特别注意：当前迁移里 scheduler 不维护真正的 pending queue，因此 `min_queue_length` 在实际语义上更接近“选当前 inflight 最少的实例”。

## 系统组成

从代码组织上看，这个闭环由三部分组成：

| 组件 | 作用 | 入口 |
| --- | --- | --- |
| Global Scheduler Runtime | 接收请求、选择实例、维护实例状态并转发 | [`vllm_omni/global_scheduler/`](./vllm_omni/global_scheduler/) |
| Diffusion Benchmark | 构造请求、压测服务、统计指标 | [`benchmarks/diffusion/`](./benchmarks/diffusion/) |
| Benchmark Orchestrator | 启动 scheduler / worker 并组织实验 | [`benchmarks/diffusion/scripts/global_instance_scheduler_v2/`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/) |

推荐的理解方式是：

- 想理解“请求是怎么被选路的”，看 runtime README
- 想理解“请求是怎么被构造和统计的”，看 benchmark README
- 想理解“实验是怎么一键跑起来的”，看 orchestrator README

## 最小启动配置

如果你的目标是先跑通一个最小闭环，推荐直接使用“启动 global scheduler，由它自动拉起 worker，再运行 benchmark”的方式，也就是：

- 一个 global scheduler
- 一个 worker
- 一份最小 scheduler 配置
- 一条 benchmark 命令

### 最小必备项

至少需要准备好以下内容：

- 可用的 Python 运行环境，并且当前环境能够导入 `vllm` / `vllm_omni`
- 当前环境里可以直接执行 `vllm`
- 一份可启动的 diffusion 模型路径
- 至少一张可用 GPU
- 一个 scheduler 配置文件

### 自动拉起语义

当前推荐的最小路径只保留自动拉起模式。

- 如果 `instances[*]` 里提供了 `launch`，那么 global scheduler 启动时会自动尝试拉起该实例
- scheduler 的自动拉起逻辑会把启动命令固定生成为 `<launch.executable> serve <launch.model> --port <endpoint_port> ...`
- 这意味着自动拉起路径面向的是 `vllm serve ...` 这类 CLI 形式
- 如果实例同时配置了 `stop`，scheduler 退出时也会自动尝试停止该实例

### 最小配置示例

下面是一个经过裁剪的最小单实例 scheduler 配置示例，适合通过 scheduler 自动拉起 worker：

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 600

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: min_queue_length

instances:
  - id: worker0
    endpoint: http://127.0.0.1:8001
    instance_type: qwen-image
    backends:
      - vllm-omni
    launch:
      executable: vllm
      model: /path/to/Qwen-Image
      args:
        - --omni
        - --diffusion-scheduler-backend
        - step_level_request_scheduler
        - --diffusion-enable-step-chunk
      env:
        CUDA_VISIBLE_DEVICES: "0"
    stop:
      executable: pkill
      args:
        - -f
        - vllm serve /path/to/Qwen-Image --port {endpoint_port}
```

这个最小配置背后的关键约束有 5 个：

- `instances[*].endpoint` 必须指向实际 worker 地址
- `launch.model` 必须和你希望启动的模型一致
- `launch.executable` 对应的命令当前必须能在环境中直接执行
- `instances[*].backends` 必须和你后续 benchmark 使用的 backend 匹配
- 如果后续切到 `short_queue_runtime`，建议请求显式携带 `estimated_cost_s`

### 最小启动方式

建议按下面 3 步执行。

#### 第 1 步：写一份最小 scheduler 配置

例如保存为 `./tmp/global_scheduler.minimal.qwen.yaml`：

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 600

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: min_queue_length

instances:
  - id: worker0
    endpoint: http://127.0.0.1:8001
    instance_type: qwen-image
    backends:
      - vllm-omni
    launch:
      executable: vllm
      model: /path/to/Qwen-Image
      args:
        - --omni
        - --diffusion-scheduler-backend
        - step_level_request_scheduler
        - --diffusion-enable-step-chunk
      env:
        CUDA_VISIBLE_DEVICES: "0"
    stop:
      executable: pkill
      args:
        - -f
        - vllm serve /path/to/Qwen-Image --port {endpoint_port}
```

#### 第 2 步：启动 global scheduler

```bash
python -m vllm_omni.global_scheduler.server \
  --config ./tmp/global_scheduler.minimal.qwen.yaml
```

启动后，scheduler 会自动拉起 `worker0`，并在 worker 的 `/v1/models` readiness probe 成功后把它变成可路由实例。
可以先做一个最基本的 smoke check：

```bash
curl http://127.0.0.1:8089/health
curl http://127.0.0.1:8089/instances
```

如果你看到 `process_state=running` 且 `healthy=true`，就可以开始发压。

#### 第 3 步：直接运行 benchmark

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8089 \
  --model /path/to/Qwen-Image \
  --backend vllm-omni \
  --task t2i \
  --dataset random \
  --num-prompts 5 \
  --max-concurrency 1 \
  --random-request-config '[{"width":1024,"height":1024,"num_inference_steps":25,"weight":1.0}]'
```

这个例子里，请求会按下面的路径流转：

- benchmark -> `http://127.0.0.1:8089`
- scheduler -> 路由到 `http://127.0.0.1:8001`
- worker -> 实际执行图像生成

如果你想改成视频最小示例，只需要同步替换下面几项：

- worker 模型路径改成视频模型
- `launch.args` 补成对应的视频并行配置
- scheduler 配置里的 `instance_type` / `backends`
- benchmark 的 `--backend v1/videos`、`--task t2v` 和请求 shape

### 脚本化实验入口

如果你的目标不是“先跑通一个最小闭环”，而是“批量跑 case / 自动汇总结果”，再使用 orchestrator 脚本和模板 YAML：

- 图像单实例模板：[`benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml)
- 视频单实例模板：[`benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml)
- 多实例模板：[`benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml)

这套方式会额外负责：

- 生成最终 YAML
- 自动启动和停止 worker
- 运行单 case 或 suite
- 归档 metrics 与 summary 结果

## 典型使用路径

根据不同目标，推荐按下面的顺序阅读和操作。

### 场景 1：先跑通一个最小闭环

1. 先读本文“最小启动配置”
2. 先按本文的“最小启动方式”启动 scheduler，并确认它自动拉起 worker 后再运行 benchmark
3. 再看 [`vllm_omni/global_scheduler/README_zh.md`](./vllm_omni/global_scheduler/README_zh.md) 理解路由语义

### 场景 2：要改路由策略或理解调度语义

重点看 global scheduler README，特别是：

- 运行模型
- 请求处理主流程
- 路由策略说明
- 生命周期与健康状态
- 配置结构

### 场景 3：要调 benchmark 流量形态或做 SLO 评估

重点看 diffusion benchmark README，特别是：

- 数据集模式
- `trace` 与 `random` 的请求字段
- warmup 语义
- `estimated_cost_s` 与 `slo_ms` 的注入方式
- `max_concurrency` 与 `request_rate` 的含义

### 场景 4：要批量跑实验或生成汇总结果

重点看 orchestrator README，特别是：

- `run_case.sh`
- `run_suite.sh`
- 环境变量覆盖规则
- `CASE_MATRIX`
- 输出目录结构

## 文档导航

下面这 4 份文档分别负责不同层级的信息，建议配合阅读。

| 文档 | 适合什么时候看 | 主要内容 |
| --- | --- | --- |
| [`README_zh.md`](./README_zh.md) | 第一次进入项目，想快速建立全局认识 | 总览、边界、最小启动路径 |
| [`vllm_omni/global_scheduler/README_zh.md`](./vllm_omni/global_scheduler/README_zh.md) | 需要理解 runtime 实现和路由语义 | API、状态管理、策略、配置结构 |
| [`benchmarks/diffusion/README_zh.md`](./benchmarks/diffusion/README_zh.md) | 需要单独使用 benchmark 或调整压测参数 | 数据集、warmup、SLO、并发与输出 |
| [`benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md) | 需要一键跑 case / suite | 模板 YAML、环境变量、运行流程、输出目录 |

## 建议的阅读顺序

如果你是第一次接手这个模块，推荐按下面的顺序阅读：

1. 本文，先建立边界和整体心智模型
2. 先按本文的“最小启动方式”启动 scheduler，并确认它自动拉起 worker 后跑通一遍
3. [`vllm_omni/global_scheduler/README_zh.md`](./vllm_omni/global_scheduler/README_zh.md)，再理解实际路由语义
4. [`benchmarks/diffusion/README_zh.md`](./benchmarks/diffusion/README_zh.md)，最后按实验目标细调 benchmark
5. 如果需要批量实验，再看 [`benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md`](./benchmarks/diffusion/scripts/global_instance_scheduler_v2/README_zh.md)

## 备注

这份 README 是一个总入口，不会重复展开所有实现和参数细节。若你需要修改实例生命周期、健康探测、路由打分、warmup profile、trace 回放或 suite 运行方式，请直接跳转到对应子目录 README。
