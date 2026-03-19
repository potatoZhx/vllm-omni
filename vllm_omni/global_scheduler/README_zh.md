# Global Scheduler 使用指南（中文）

本目录提供 vLLM-Omni 的 global scheduler 代理服务。
它暴露 OpenAI 兼容入口，并把请求路由到多个上游 vLLM 实例。

主模块：

- `vllm_omni/global_scheduler/server.py`

## 1. 快速开始

### 1.1 创建调度配置

创建 `global_scheduler.yaml`：

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 1800
  instance_health_check_interval_s: 5.0
  instance_health_check_timeout_s: 1.0

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: short_queue_runtime  # fcfs | min_queue_length | round_robin | short_queue_runtime | estimated_completion_time
    runtime_profile_path: ./runtime_profile.json

benchmark:
  worker_ids: [worker-0, worker-1]
  worker_ready_timeout_s: 600
  model: Qwen/Qwen-Image
  backend: vllm-omni
  task: t2i
  dataset: trace
  max_concurrency: 20
  auto_stop: true

instances:
  - id: worker-0
    endpoint: http://127.0.0.1:9001
    instance_type: qwen-image-tp2
    numa_node: 0
    backends: [vllm-omni, openai]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "0,1"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port {endpoint_port}"]
  - id: worker-1
    endpoint: http://127.0.0.1:9002
    instance_type: wan-video-tp2
    numa_node: 1
    backends: [v1/videos]
    launch:
      executable: vllm
      model: Wan/Wan2.2
      args: ["--omni", "--max-concurrency", "2"]
      env:
        CUDA_VISIBLE_DEVICES: "2,3"
```

说明：

- `policy.baseline.runtime_profile_path`
  - 给 `short_queue_runtime` / `estimated_completion_time` 提供 profiling JSON
- `instances[].instance_type`
  - 与 runtime profile 中的 `instance_type` 关联，用于选择实例类型专属耗时画像
- `instances[].numa_node`
  - 若机器装有 `numactl`，启动时会自动为该实例加上对应的 NUMA 绑核参数
- `launch.args`
  - 只写额外参数；`vllm serve <model> --port <endpoint_port>` 由 scheduler 自动补齐
- `stop.args`
  - 支持占位符：`{instance_id}`、`{endpoint}`、`{endpoint_host}`、`{endpoint_port}`
- `benchmark`
  - 供 `scripts/run_global_scheduler_benchmark*.sh` 读取的基准测试配置块

### 1.2 启动 global scheduler

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
```

scheduler 会监听配置中的 `http://<host>:<port>`（默认 `8089`）。

当前实现的重要行为：

- 该命令会启动 scheduler 服务本身。
- 若某个实例配置了 `launch`，服务启动阶段会自动对它执行一次 `start`。
- auto-start 后实例会先进入 `process_state=running`，但通常会在探针成功前保持 `healthy=false`、`routable=false`。
- 若某个实例没有 `launch`，scheduler 不会替你拉起它。
- 服务退出时，若实例配置了 `stop`，scheduler 会 best-effort 执行一次 stop。

### 1.3 手动触发探针或生命周期操作

如果你想立刻刷新可路由状态，先执行：

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

如果实例没有自动拉起、需要重启，或你想手动控制进程：

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-1/restart
```

### 1.4 检查可路由状态

```bash
curl -sS http://127.0.0.1:8089/health
curl -sS http://127.0.0.1:8089/instances
```

至少要有一个实例满足：

- `enabled=true`
- `healthy=true`
- `draining=false`
- `process_state=running`
- `routable=true`

### 1.5 发送一条请求做冒烟验证

```bash
curl -sS http://127.0.0.1:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image",
    "messages": [{"role": "user", "content": "a cute orange cat"}],
    "extra_body": {
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 20
    }
  }'
```

## 2. 运行时 API

### 2.1 请求入口

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `POST /v1/videos`

后端路由说明：

- `/v1/chat/completions` 对应 backend `vllm-omni`
- `/v1/images/generations` 对应 backend `openai`
- `/v1/videos` 对应 backend `v1/videos`
- `instances[].backends` 控制实例可接收哪些 backend
- 若省略或留空 `instances[].backends`，实例默认兼容全部 backend

调度器会尽量从请求里提取以下元数据参与选路：

- `width`
- `height`
- `num_frames`
- `num_inference_steps`

提取来源：

- chat/images JSON 里的 `extra_body`
- chat/images JSON 顶层字段
- OpenAI image 的 `size`
- `/v1/videos` multipart form fields

响应头包含：

- `X-Routed-Instance`: 被选中的实例 id
- `X-Route-Reason`: 选路原因
- `X-Route-Score`: 选路分数（字符串化浮点数）

### 2.2 健康与实例状态

- `GET /health`
  - 返回 `status`、`instance_count`、`version`
  - `checks` 当前包含：
    - `config_loaded`
    - `has_instances`
- `GET /instances`
  - 返回每个实例的生命周期和运行时快照

`/instances` 的每个实例当前会包含：

- `id`
- `endpoint`
- `backends`
- `enabled`
- `healthy`
- `draining`
- `process_state`
- `last_operation`
- `last_operation_ts_s`
- `last_operation_error`
- `last_check_ts_s`
- `last_error`
- `log_path`
- `routable`
- `queue_len`
- `inflight`
- `ewma_service_time_s`

其中：

- `routable = enabled && healthy && !draining && process_state == "running"`
- `log_path` 默认为 `./logs/global_scheduler/<instance_id>.log`
  - 可通过环境变量 `GLOBAL_SCHEDULER_LOG_DIR` 修改

### 2.3 生命周期操作 API（当前实现）

- `POST /instances/{id}/disable`
- `POST /instances/{id}/enable`
- `POST /instances/{id}/stop`
- `POST /instances/{id}/start`
- `POST /instances/{id}/restart`
- `POST /instances/reload`
- `POST /instances/probe`

示例：

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/disable
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/enable
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/stop
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/restart
curl -sS -X POST http://127.0.0.1:8089/instances/reload
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

说明：

- `disable/enable`
  - 只改变 scheduler 的路由可用性，不直接管理进程
- `start/restart`
  - 依赖 `instances[].launch`
- `stop/restart`
  - 依赖 `instances[].stop`
- `start`
  - 对“已经 running 且上一次操作也是 start”的实例是幂等的，不会重复拉起进程
- `reload`
  - 需要通过 `--config` 启动，且会重新加载 YAML、重建 policy、同步实例清单

## 3. 路由策略

通过 YAML 配置：

- `policy.baseline.algorithm=fcfs`
- `policy.baseline.algorithm=min_queue_length`
- `policy.baseline.algorithm=round_robin`
- `policy.baseline.algorithm=short_queue_runtime`
- `policy.baseline.algorithm=estimated_completion_time`

相关参数：

- `scheduler.tie_breaker`
  - `random` 或 `lexical`
- `scheduler.ewma_alpha`
  - 实例 EWMA 服务时间平滑系数 `(0, 1]`
- `policy.baseline.runtime_profile_path`
  - runtime profile JSON 路径
- `instances[].instance_type`
  - profile 命中时使用的实例类型标签

策略差异：

- `fcfs`
  - 选第一个可用实例；同分时按 tie-breaker
- `min_queue_length`
  - 选 `queue_len` 最小的实例
- `round_robin`
  - 在可用实例之间轮转
- `short_queue_runtime`
  - 选“估算剩余队列总运行时间”最小的实例
  - 会累加 waiting requests 的 profile/EWMA 估时，并加上 `inflight * ewma_service_time_s`
- `estimated_completion_time`
  - 选“当前请求在该实例上估算完成时间”最小的实例
  - 当前实现近似为 `queue_len * 当前请求估时 + 当前请求估时`

补充说明：

- `short_queue_runtime` 和 `estimated_completion_time` 会在 profile 缺失时回退到实例 EWMA
- profile JSON 需要 `profiles` 数组，记录里使用 `latency_ms`
- `--max-concurrency` 会被 scheduler 用来推导实例并发容量，但不会透传给实际的 `vllm serve` 子进程

## 4. 错误语义

请求转发路径和大多数生命周期操作返回统一错误体：

```json
{
  "error": {
    "code": "GS_...",
    "message": "...",
    "request_id": "..."
  }
}
```

常见错误码：

- `GS_NO_ROUTABLE_INSTANCE` (503)
- `GS_UPSTREAM_TIMEOUT` (502)
- `GS_UPSTREAM_NETWORK_ERROR` (502)
- `GS_UPSTREAM_HTTP_ERROR` (上游状态码)
- `GS_LIFECYCLE_CONFLICT` (409)
- `GS_LIFECYCLE_UNSUPPORTED` (400)
- `GS_LIFECYCLE_EXEC_ERROR` (502)
- `GS_UNKNOWN_INSTANCE` (404)

补充说明：

- `reload`、`enable`、`disable` 等管理接口的部分错误仍可能返回 FastAPI 默认错误体，而不是 `GS_*`

## 5. 通过 scheduler 做 benchmark

将 `--base-url` 指向 scheduler 即可覆盖完整链路。

示例：

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8089 \
  --model Qwen/Qwen-Image \
  --task t2i \
  --dataset vbench \
  --num-prompts 20 \
  --max-concurrency 4
```

完整路径：

- benchmark client -> global scheduler -> 选中的上游实例

仓库内辅助脚本：

- `scripts/run_global_scheduler_benchmark.sh`
- `scripts/run_global_scheduler_benchmark_one_shell.sh`
- `scripts/run_global_scheduler_benchmark_one_shell_cleanup.sh`

这些脚本会读取同一个 YAML 里的 `benchmark` 配置块。

## 6. 故障排查

### 6.1 `GS_NO_ROUTABLE_INSTANCE`

检查：

- `GET /instances` 至少有一个实例满足：
  - `enabled=true`
  - `healthy=true`
  - `draining=false`
  - `process_state=running`
  - `routable=true`
- 配置中的 endpoint 可达（`http://host:port`，且不包含 path）
- 如果服务刚启动且实例已被 auto-start，先执行一次 `POST /instances/probe`

### 6.2 频繁出现 `GS_UPSTREAM_TIMEOUT`

检查：

- `server.request_timeout_s` 是否足够大
- 上游是否过载（`inflight` 接近实例并发上限）
- 健康探针超时是否过于激进
- `short_queue_runtime` / `estimated_completion_time` 是否缺少 runtime profile，导致估时过粗

### 6.3 启动时配置校验失败

常见原因：

- `instances[].id` 重复
- `policy.baseline.algorithm` 非法
- `policy.baseline.runtime_profile_path` 为空字符串
- endpoint 格式错误（必须是 `http://host:port`，且不能带 path）
- `instances[].backends` 含非法 backend
- `instances[].instance_type` 为空字符串
- `instances[].numa_node < 0`
- `launch/stop` 结构化配置不合法

### 6.4 生命周期操作成功了，但实例仍不可路由

检查：

- `start` 之后实例可能只是 `process_state=running`，但 HTTP 还没 ready
- 查看 `last_error`
  - 常见值如 `awaiting_http_ready_after_start`
  - 或 `awaiting_probe_after_start`
- 查看 `log_path` 对应日志，确认上游服务是否真的启动成功
