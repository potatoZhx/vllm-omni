# Global Scheduler Policies

- `fcfs`: 选择配置顺序中的第一个可用实例；如果都忙则按 tie-breaker 回退。
- `min_queue_length`: 选择 `RuntimeStats.inflight + RuntimeStats.queue_len` 最小的实例，也就是当前总未完成请求数最少的实例，不做时长估算。
- `round_robin`: 在实例间轮转分发，请求尽量均衡落到不同实例。
- `short_queue_runtime`: 估算当前总未完成工作量，近似为 `inflight * EWMA服务时长 + waiting_requests 的预计总运行时长`，选择预计剩余工作量最小的实例。
- `estimated_completion_time`: 估算当前队列加上新请求后的完成时间，选择最早完成的实例。

平分时统一使用 `scheduler.tie_breaker`，当前支持 `random` 和 `lexical`。
