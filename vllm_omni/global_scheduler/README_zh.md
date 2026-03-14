# Global Scheduler 使用指南（中文）

本目录提供 vLLM-Omni 的 global scheduler 代理服务。  
它提供 OpenAI 兼容入口，并把请求路由到多个上游 vLLM 实例。

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
    algorithm: fcfs  # fcfs | min_queue_length | round_robin | short_queue_runtime | estimated_completion_time

instances:
  - id: worker-0
    endpoint: http://127.0.0.1:9001
    backends: [vllm-omni, openai]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "0,1"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port 9001"]
  - id: worker-1
    endpoint: http://127.0.0.1:9002
    backends: [v1/videos]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "2,3"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port 9002"]
```

### 1.2 启动 global scheduler

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
```

scheduler 会监听配置中的 `http://<host>:<port>`（默认 `8089`）。

重要说明：

- 该命令只会启动 scheduler 服务本身。
- 启动时不会自动拉起所有上游 worker。
- 需要通过 lifecycle API（`/instances/{id}/start|stop|restart`）由 scheduler 控制实例进程。

### 1.3 通过 scheduler 启动 worker

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-1/start
```

### 1.4 检查可路由状态

```bash
curl -sS http://127.0.0.1:8089/instances
```

至少要有一个实例满足：

- `enabled=true`
- `healthy=true`
- `draining=false`
- `process_state=running`

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
- 通过 `instances[].backends` 控制实例可接收的 backend。
- 若省略或留空 `instances[].backends`，实例默认兼容全部 backend。

响应头包含：

- `X-Routed-Instance`: 被选中的实例 id
- `X-Route-Reason`: 选路原因
- `X-Route-Score`: 选路分数（字符串化浮点数）

### 2.2 健康与实例状态

- `GET /health`: 服务健康与配置状态
- `GET /instances`: 实例生命周期和运行时状态快照

示例：

```bash
curl -sS http://127.0.0.1:8089/health
curl -sS http://127.0.0.1:8089/instances
```

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

- `start/restart` 依赖 `instances[].launch` 配置。
- `stop/restart` 依赖 `instances[].stop` 配置。
- `reload` 需要通过 `--config` 启动（模块入口默认支持 reload loader）。

## 3. 路由策略

通过 YAML 配置：

- `policy.baseline.algorithm=fcfs`
- `policy.baseline.algorithm=min_queue_length`
- `policy.baseline.algorithm=round_robin`
- `policy.baseline.algorithm=short_queue_runtime`
- `policy.baseline.algorithm=estimated_completion_time`

相关参数：

- `scheduler.tie_breaker`: `random` 或 `lexical`
- `scheduler.ewma_alpha`: EWMA 平滑系数 `(0, 1]`
- 实例路由并发上限由 `instances[].launch.args` 推导
  - 推荐使用 `--max-concurrency`

## 4. 错误语义

global scheduler 返回统一错误体：

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
- `GS_UPSTREAM_HTTP_ERROR` (透传上游状态码)
- `GS_LIFECYCLE_CONFLICT` (409)
- `GS_LIFECYCLE_UNSUPPORTED` (400)
- `GS_LIFECYCLE_EXEC_ERROR` (502)

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

## 6. 故障排查

### 6.1 `GS_NO_ROUTABLE_INSTANCE`

检查：

- `GET /instances` 至少有一个实例满足：
  - `enabled=true`
  - `healthy=true`
  - `draining=false`
  - `process_state=running`
- 配置中的 endpoint 可达（`http://host:port`，且不包含 path）

### 6.2 频繁出现 `GS_UPSTREAM_TIMEOUT`

检查：

- `server.request_timeout_s` 是否足够大
- 上游是否过载（`inflight` 接近 `max_concurrency`）
- 健康探针超时设置是否过于激进

### 6.3 启动时配置校验失败

常见原因：

- `instances[].id` 重复
- `policy.baseline.algorithm` 非法
- endpoint 格式错误（必须是 `http://host:port`）
- `launch/stop` 结构化配置不合法

## 7. 当前限制

- 不支持动态 SP 调度。
- 当前未提供 `/metrics`。
- 目前仅为本地命令模板控制，生产环境建议接入编排器原生执行器。
