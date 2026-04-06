# vLLM-Omni Global Scheduler

这个目录提供了 `v18-base` 上用于多实例 diffusion online serving 的
global scheduler 实现。

这轮迁移里的实现是刻意收窄的：

- 它是全局请求路由器，不是 scheduler 侧等待队列
- 请求到来后会立即选 worker 并转发
- 它只维护实例级 runtime bookkeeping，用于策略打分
- worker 内部的 step-level 执行和本地调度仍由 worker 自己负责

整体目标是：在不引入过高复杂度的前提下，让 `v18-base` 先具备一个可运行、
可验证、可做 benchmark 的最小 global scheduler 闭环。

## 本轮迁移范围

本轮支持的全局策略：

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

支持的 backend 家族：

- `vllm-omni`
- `openai`
- `v1/videos`

本轮明确不做：

- scheduler 侧等待 / admission blocking
- 全局容量 gating
- `fcfs`
- `estimated_completion_time`
- 实例内高级策略，例如 `sjf`、`p95-first`、`guarded`、`fusion`
- worker 侧的 `slo_target_ms`、`deadline_ts` 等字段

## 运行模型

当前实现维护两类状态：

1. 静态实例配置
2. 已经路由出去请求的 runtime bookkeeping

这会带来一个重要结论：

- 这个实现里没有 scheduler 侧 pending queue
- `queue_len` 当前始终为 `0`
- 真实有意义的动态信号主要是 `inflight`、`outstanding_runtime_s` 和 `ewma_service_time_s`

因此：

- `min_queue_length` 当前本质上等价于按 `inflight` 数量选
- `round_robin` 不看负载，只按轮转选
- `short_queue_runtime` 在请求携带代价信息时，按累计未完成 runtime 选

## 请求处理主流程

每个进入 scheduler 的请求都会经历：

1. 从请求里抽取 `RequestMeta`
2. 按 backend 兼容性和生命周期 routability 过滤实例
3. 基于当前 runtime snapshot 执行全局策略
4. 为选中的实例做 runtime reservation
5. 把原始 HTTP 请求转发到对应 worker
6. 当上游返回或失败时，释放 reservation 并更新 EWMA

这里不会发生“先在 global scheduler 里排队，等容量释放后再发”的行为。

## 对外 HTTP 接口

健康检查和实例视图：

- `GET /health`
- `GET /instances`
- `POST /instances/reload`
- `POST /instances/probe`

生命周期操作：

- `POST /instances/{id}/enable`
- `POST /instances/{id}/disable`
- `POST /instances/{id}/start`
- `POST /instances/{id}/stop`
- `POST /instances/{id}/restart`

代理转发入口：

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `POST /v1/videos`

每次已经做出选路的响应都会附带：

- `X-Routed-Instance`
- `X-Route-Reason`
- `X-Route-Score`

## Backend 映射关系

scheduler 里三类入口和 backend 名称的对应关系是：

- `/v1/chat/completions` -> `vllm-omni`
- `/v1/images/generations` -> `openai`
- `/v1/videos` -> `v1/videos`

每个实例可以通过 `backends` 声明允许接收的 backend 列表。

如果某个实例的 `backends` 为空，则默认认为它兼容所有支持的 backend。

## 路由时可见的请求元数据

当前 scheduler 会抽取这些字段：

- `width`
- `height`
- `num_frames`
- `num_inference_steps`
- `estimated_cost_s`
- `slo_ms`

抽取方式：

- 对 `/v1/chat/completions` 和 `/v1/images/generations`，既可以来自顶层字段，也可以来自 `extra_body`
- 对图像类 OpenAI 请求，还会解析 `size="WxH"`
- 对 `/v1/videos`，当前从 multipart form fields 里取元数据

当前真正参与路由的重点字段：

- `estimated_cost_s` 是目前最有用的 scheduler-visible cost 信号
- `slo_ms` 会被解析并保存在 `RequestMeta.extra`，但当前策略并不会消费它

## 路由策略说明

### `min_queue_length`

它选择下列值最小的实例：

- `inflight + queue_len`

由于当前迁移里 `queue_len` 始终为 `0`，所以它等价于：

- 选择当前 active routed request 最少的实例

### `round_robin`

它按轮转顺序在 routable 实例之间选路。

特点：

- 不看请求 shape
- 不看 runtime 代价
- 返回的 `score` 是被选实例当前的 `inflight`

### `short_queue_runtime`

它选择 `outstanding_runtime_s` 最小的实例。

但有一个关键细节：

- 如果当前请求没有携带 `estimated_cost_s`，这个策略会直接 fallback 到 `min_queue_length`

所以如果你希望这个策略真正体现 runtime-aware 选路，调用方或 benchmark 最好显式提供
`estimated_cost_s`。

## Runtime 估算逻辑

每个已路由请求占用多少 reserved runtime，由 `RuntimeEstimator` 计算。

优先级顺序是：

1. 请求里的 `estimated_cost_s`
2. runtime profile 精确匹配
3. runtime profile 按相邻 `steps` 插值
4. 实例级 EWMA 回退

这个估算值会写进 `outstanding_runtime_s`，用于后续请求的打分。

需要特别注意：

- runtime bookkeeping 本身可以利用 runtime profile / EWMA 回退
- 但 `short_queue_runtime` 这个策略在“当前请求没有 `estimated_cost_s`”时，选路阶段仍然会回退到 `min_queue_length`

## Runtime Profile 文件格式

如果配置了 `policy.baseline.runtime_profile_path`，scheduler 会加载一个 JSON 文件。

顶层结构要求：

- 必须是一个对象
- 必须包含 `profiles` 数组

每个 profile 支持的字段：

- `instance_type`
- `width`
- `height`
- `num_frames`
- `steps`
- `latency_ms`

示例：

```json
{
  "profiles": [
    {
      "instance_type": "wan-video-tp2",
      "width": 1280,
      "height": 720,
      "num_frames": 16,
      "steps": 50,
      "latency_ms": 8210
    }
  ]
}
```

内部索引 key 为：

- `(instance_type, width, height, num_frames, steps)`

内部统一以秒存储。

## 生命周期与健康状态

每个实例都维护以下状态：

- `enabled`
- `healthy`
- `draining`
- `process_state`

一个实例只有同时满足以下条件时才可路由：

- `enabled == true`
- `healthy == true`
- `draining == false`
- `process_state == "running"`

### 健康探测

scheduler 会启动后台 probe loop。

在 `start` 或 `restart` 之后：

- 会优先探测 `GET /v1/models`
- 只有 worker 能返回至少一个 model，才算 ready

进入稳定运行阶段之后：

- 会探测 `GET /health`

连续失败达到 `server.instance_health_check_failures_before_unhealthy` 后，
实例会被标记为 unhealthy。

### `disable` 与 `stop` 的区别

`disable`：

- 不再接收新路由
- 实例会进入 draining
- 不会停止进程

`stop`：

- 会执行配置好的 stop 命令
- 实例会变成不可用

`enable`：

- 重新允许路由
- 立即把实例标记为 healthy

`start` / `restart`：

- 执行配置好的 launch 命令
- 重新启用实例
- 但在 readiness probe 成功前会先保持 unhealthy

### 配置热重载

`POST /instances/reload` 会重新读取启动时传入的同一份 YAML。

reload 的行为：

- 新实例会被加入
- 已有实例的配置会被刷新
- 被删除的实例如果没有 inflight 请求，会立即移除
- 被删除的实例如果仍有 inflight 请求，会先变成 draining，等 drain 完再删除

## 进程控制

生命周期操作依赖 `LocalProcessController`。

启动命令会被构造成：

- `<launch.executable> serve <launch.model> --port <endpoint_port> ...launch.args`

停止命令则是：

- `stop.executable` 加 `stop.args`

`stop.args` 当前支持这些占位符：

- `{instance_id}`
- `{endpoint}`
- `{endpoint_host}`
- `{endpoint_port}`

实例日志路径：

- 默认写到 `./logs/global_scheduler`
- 可以通过环境变量 `GLOBAL_SCHEDULER_LOG_DIR` 改掉

NUMA 行为：

- 如果系统里有 `numactl`，并且实例配置了 `numa_node`
- 启动命令会自动加上 `numactl --cpunodebind=... --membind=...`

## 配置结构

根配置分为：

- `server`
- `scheduler`
- `policy`
- `benchmark`
- `instances`

真正被 runtime server 消费的是：

- `server`
- `scheduler`
- `policy`
- `instances`

`benchmark` 段主要是为了和 benchmark / orchestration 脚本共用一份 YAML，方便放在一起管理；
它不是当前 runtime 路由主路径依赖的配置。

### `server`

主要字段：

- `host`
- `port`
- `request_timeout_s`
- `instance_health_check_interval_s`
- `instance_health_check_timeout_s`
- `instance_health_check_failures_before_unhealthy`

### `scheduler`

主要字段：

- `tie_breaker`：`random` 或 `lexical`
- `ewma_alpha`

### `policy.baseline`

主要字段：

- `algorithm`
- `runtime_profile_path`

支持的算法：

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

### `instances[]`

主要字段：

- `id`
- `endpoint`
- `instance_type`
- `numa_node`
- `backends`
- `launch`
- `stop`

`endpoint` 必须是：

- `http://host:port`

不能带 path。

## 配置示例

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
    algorithm: short_queue_runtime
    runtime_profile_path: ./runtime_profile.json

instances:
  - id: worker0
    endpoint: http://127.0.0.1:8001
    instance_type: qwen-image
    backends:
      - vllm-omni
      - openai
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args:
        - --omni
        - --diffusion-scheduler-backend
        - step_level_request_scheduler
        - --diffusion-enable-step-chunk
    stop:
      executable: pkill
      args:
        - -f
        - vllm serve Qwen/Qwen-Image --port {endpoint_port}
```

## 启动方式

可以这样启动：

```bash
python -m vllm_omni.global_scheduler.server --config /path/to/config.yaml
```

最基本的 smoke 检查：

```bash
curl http://127.0.0.1:8089/health
curl http://127.0.0.1:8089/instances
```

## 运维注意点

- 只要实例提供了 `launch`，scheduler 启动时就会自动尝试 start
- 只要实例提供了 `stop`，scheduler 退出时就会自动尝试 stop
- 生命周期操作是按实例串行化的
- 如果正在 reload 或另一个 lifecycle 操作还没结束，会拒绝新的冲突操作

## 已知限制

- 这轮迁移没有 scheduler 侧等待队列
- 这轮迁移没有全局容量约束
- `queue_len` 当前不是一个真实 waiting metric
- `short_queue_runtime` 要想真正体现 runtime-aware 选路，最好让请求显式携带 `estimated_cost_s`
- `/v1/videos` 的元数据提取目前基于 multipart form fields
- 配置里的 `benchmark` 段虽然会被校验，但当前 runtime 路由主路径并不消费它
