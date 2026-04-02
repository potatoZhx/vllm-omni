# vLLM-Omni Global Scheduler

这是 `v18-base` 上的最小版 global scheduler 迁移实现。

本轮只保留 3 个全局策略：

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

当前实现边界：

- global scheduler 只负责即时选路和转发
- 不在 global scheduler 侧维护等待队列
- step-level 执行仍由 worker 启动参数控制
- `short_queue_runtime` 优先消费请求携带的 `estimated_cost_s`
