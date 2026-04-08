# `v16-base` -> `v18-base` Merge 冲突整理

## 背景

- 目标分支：`v18-base`
- 待合并分支：`v16-base`
- 检查方式：在临时 worktree 中执行 `git merge --no-commit --no-ff v16-base`
- 本文不是 raw marker 全量转存，而是把每个冲突文件中双方“分别改了什么”整理成可决策的信息

## 总览

- 真实冲突文件数：`18`
- `UU`：`14`
- `AA`：`3`
- `DU`：`1`

冲突热点：

- `vllm_omni/diffusion`：`6`
- `vllm_omni/entrypoints`：`4`
- `benchmarks/diffusion`：`4`
- `tests`：`3`
- 仓库配置：`1`

## 配置类冲突

### `.pre-commit-config.yaml`

- 冲突类型：`UU`
- `v18-base`：保留启用状态的 pre-commit 配置，包含 `pre-commit-hooks`、`ruff`、`typos`、`actionlint`、`signoff-commit`
- `v16-base`：将 pre-commit 整体置为“禁用”，文件前面只保留两行说明注释，下面完整配置全部被注释掉
- 决策重点：是沿用上游当前启用的 pre-commit 体系，还是继续保持你本地分支里“默认禁用 hooks”的开发方式

## Benchmark / 文档冲突

### `benchmarks/diffusion/backends.py`

- 冲突类型：`UU`
- `v18-base`：
  - `/v1/videos` 走异步 job 模式
  - `POST` 只拿 `id/status`
  - 后续 `GET /v1/videos/{id}` 轮询直到 `completed/failed`
  - 再 `GET /content` 拉二进制内容
  - 结束时还尝试 `DELETE` 清理 job
  - backend mapping 改成按任务类型分层：`2i -> {vllm-omni, openai}`，`2v -> {v1/videos}`
- `v16-base`：
  - 仍把 `/v1/videos` 当作同步返回结果的普通 backend
  - `POST` 后直接拿响应体当结果
  - backend mapping 是扁平结构：`vllm-omni`、`openai`、`v1/videos`
- 决策重点：benchmark 端到底要匹配上游的视频异步 API 语义，还是继续匹配你本地旧版同步 benchmark 调用路径

### `benchmarks/diffusion/diffusion_benchmark_serving.py`

- 冲突类型：`UU`
- `v18-base`：
  - 先根据 `task` 推导任务类型：`2i` / `2v`
  - 再根据任务类型校验合法 backend
  - 不合法时直接报错
  - warmup 时按 `requests_list` 动态构造请求，并支持 `warmup_num_inference_steps`
- `v16-base`：
  - 遇到不兼容的 task/backend 组合时，不报错，而是自动重写 backend
  - `VIDEO_TASKS` 强制改成 `v1/videos`
  - `IMAGE_TASKS` 遇到 `v1/videos` 时强制改成 `vllm-omni`
  - warmup 直接遍历预构建好的 `warmup_requests`
- 决策重点：保留“强校验显式失败”还是“自动纠正 backend”；同时 warmup 逻辑要选哪一套

### `benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md`

- 冲突类型：`AA`
- 双方都新增了同名文档
- 差异主要是标题编号：
  - `v18-base`：`# 6 / # 7 / # 8`
  - `v16-base`：`# 5 / # 6 / # 7`
- 决策重点：保留哪一版目录编号

### `benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md`

- 冲突类型：`AA`
- 双方都新增了同名文档
- 差异有两类：
  - 标题编号差异：`v18-base` 用 `6/7/8`，`v16-base` 用 `5/6/7`
  - 文末说明文字差异：
    - `v18-base`：`Wan2.2 serving performance reference`
    - `v16-base`：误写成 `Qwen-Image serving performance reference`
- 决策重点：基本应以 `v18-base` 的文末说明为准，再决定目录编号

## 测试冲突

### `tests/diffusion/test_diffusion_request.py`

- 冲突类型：`AA`
- `v18-base`：
  - 新增随机 seed 相关测试
  - 核心断言是：未显式设置 seed 时，请求会自动分配 seed，且不同请求 seed 不同
  - 带 `pytest.mark.core_model`
- `v16-base`：
  - 新增 stage1 / scheduler 默认字段测试
  - 断言 `arrival_time`、`request_state`、`executed_steps`、`max_steps_this_turn`、`dispatch_epoch`、`estimated_cost_s`、`deadline_ts`、`primary_request_id`
- 决策重点：这两个测试语义并不冲突，最终大概率都应保留

### `tests/entrypoints/test_async_omni_diffusion_config.py`

- 冲突类型：`UU`
- `v18-base`：
  - 以 `AsyncOmniEngine._create_default_diffusion_stage_cfg` 和 CLI helper 为测试入口
  - 关注点是 cache 配置、`ulysses_degree`、`ulysses_mode`、CLI 参数透传
- `v16-base`：
  - 以 `AsyncOmni._create_default_diffusion_stage_cfg` 为测试入口
  - 关注点扩展到 instance scheduler 全家桶参数
  - 覆盖 step chunk / chunk preemption / runtime profile / policy 默认值与合法性
- 决策重点：测试入口已经分叉为 `AsyncOmniEngine` 与 `AsyncOmni` 两套 builder；需要先统一生产代码入口，再决定保留哪套测试组织方式

### `tests/entrypoints/test_omni_stage_diffusion_config.py`

- 冲突类型：`DU`
- `v18-base`：该文件已删除
- `v16-base`：仍在测试 `_build_od_config`，重点覆盖：
  - `cache_backend`
  - `cache_config`
  - `vae_use_slicing`
  - `instance_scheduler_policy`
  - `instance_scheduler_slo_target_ms`
  - `instance_scheduler_p95_first_base_ms`
  - `instance_runtime_profile_path/name`
- 决策重点：如果 `v18-base` 已不再走 `_build_od_config` 这条路径，则该测试应迁移而不是直接恢复原文件

## Diffusion 核心冲突

### `vllm_omni/diffusion/request.py`

- 冲突类型：`UU`
- 这是一个“表面小、语义不小”的冲突
- `v18-base`：
  - 引入 `random`
  - 在 `__post_init__` 中，当 `generator` 和 `seed` 都为空时自动分配随机 seed
- `v16-base`：
  - 引入 `time`
  - 为请求对象增加大量 scheduler 生命周期字段：
    - `arrival_time`
    - `first_enqueue_time`
    - `first_dispatch_time`
    - `last_dispatch_time`
    - `last_preempted_time`
    - `completion_time`
    - `failure_time`
    - `aborted_time`
    - `request_state`
    - `executed_steps`
    - `max_steps_this_turn`
    - `dispatch_epoch`
    - `estimated_cost_s`
    - `deadline_ts`
    - `scheduler_force_run_to_completion`
    - `scheduler_chunk_budget_steps`
    - `primary_request_id`
- 决策重点：最终应同时保留 `auto-seed` 与 scheduler 生命周期字段；这个文件的 marker 只是 import 冲突，但语义上是 merge 关键前提

### `vllm_omni/diffusion/executor/multiproc_executor.py`

- 冲突类型：`UU`
- `v18-base`：
  - 根据 `num_gpus` 初始化 broadcast queue
  - `broadcast_handle` 来自 `_init_broadcast_queue`
- `v16-base`：
  - 显式创建 `Stage1Scheduler`
  - `scheduler.initialize(self.od_config)`
  - `broadcast_handle` 改由 scheduler 提供
- 决策重点：scheduler 的拥有者是谁
  - 上游：executor 自己维护 broadcast queue
  - 本地：executor 持有 scheduler，scheduler 再向下发广播句柄

### `vllm_omni/diffusion/worker/diffusion_model_runner.py`

- 冲突类型：`UU`
- `v18-base`：
  - 新增 `_record_peak_memory`
  - 记录当前请求的 `max_memory_reserved / max_memory_allocated`
  - 输出 pool overhead 日志
- `v16-base`：
  - 引入 step-chunk 生命周期 API：
    - `prepare_generation`
    - `step_generation`
    - `finalize_generation`
    - `abort_generation`
  - 使用 `DiffusionRequestContext` 追踪 active contexts
- 决策重点：上游新增的是观测性，`v16-base` 新增的是执行模型；大概率需要把 `_record_peak_memory` 嵌入新的分步执行路径，而不是二选一

### `vllm_omni/diffusion/diffusion_engine.py`

- 冲突类型：`UU`
- `v18-base`：
  - 仍偏 run-to-completion 路径
  - 维持 `RequestScheduler / SchedulerInterface`
  - 提供 profiling 入口
- `v16-base`：
  - 引入 `RuntimeProfileEstimator`
  - 请求执行改成 chunk-aware loop
  - 每轮先 `request.max_steps_this_turn = self._plan_chunk_budget(request)`
  - 调用 `add_req_and_wait_for_response`
  - 汇总 chunk metrics
  - 当 `finished=False` 时继续推进
  - 收尾时清理 `scheduler_force_run_to_completion` / `scheduler_chunk_budget_steps`
  - 增加 `estimate_waiting_queue_len` / `estimate_scheduler_load` / `start_profile`
- 决策重点：这是整次 merge 最核心的冲突之一，实质是“上游执行/观测模型”与“本地 scheduler + step chunk 控制流”的整合

### `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`

- 冲突类型：`UU`
- `v18-base`：
  - 引入更细粒度的 helper 结构：
    - `_extract_prompts`
    - `_prepare_generation_context`
    - `prepare_encode`
    - `_build_denoise_kwargs`
    - `_decode_latents`
  - 更像是一次 pipeline 内部结构整理
- `v16-base`：
  - 引入 scheduler state capture / restore
  - `prepare_generation` 直接返回 `DiffusionRequestContext`
  - 新增 `step_generation` / `finalize_generation`
  - `forward` 同时支持 step-chunk 和 run-to-completion
- 决策重点：上游是“把生成逻辑拆成 helper”，本地是“把生成逻辑改造成可暂停/恢复”。正确 merge 路径通常是保留 `v18-base` 的 helper 分层，再把 `v16-base` 的 context 生命周期嵌进去

### `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`

- 冲突类型：`UU`
- `v18-base`：
  - 有完整 `DEBUG_PERF` 计时逻辑
  - 统计 text encode / latent prep / denoise / decode / pipeline wall time
  - 带 progress bar
  - 最终返回 `stage_durations`
- `v16-base`：
  - 和 Qwen-Image 一样，新增 `DiffusionRequestContext`
  - `prepare_generation` / `step_generation` / `finalize_generation` / `forward`
  - 支持 chunk-based 执行与恢复 scheduler state
- 决策重点：这里的真正矛盾不是功能互斥，而是两边都深改了主 denoise loop；需要把上游 PERF instrumentation 移植到新的 step-chunk 生命周期里

## Entrypoint / OpenAI 接口冲突

### `vllm_omni/entrypoints/async_omni.py`

- 冲突类型：`UU`
- `v18-base`：
  - 冲突点附近是 comprehension-stage 相关流程
- `v16-base`：
  - 在类中加入 `_create_default_diffusion_stage_cfg`
  - 这个 builder 不只是 cache / slicing / parallel config
  - 还扩展了 instance scheduler 参数、runtime profile 参数、step chunk 参数、chunk preemption 参数
- 决策重点：`v16-base` 这段 builder 基本是整个调度功能进入配置层的入口，后续需要和 `v18-base` 的类结构一起重新安放

### `vllm_omni/entrypoints/async_omni_diffusion.py`

- 冲突类型：`UU`
- `v18-base`：在该位置放了 “Public generate API” 的章节注释
- `v16-base`：在初始化阶段新增日志，打印 `model` 和 `diffusion_engine_max_concurrency`
- 决策重点：这是轻量冲突，后续几乎可以同时保留

### `vllm_omni/entrypoints/openai/api_server.py`

- 冲突类型：`UU`
- 冲突块 1：
  - `v16-base` 增加 `_resolve_request_id`、`_log_request_arrival`、`_log_request_finish_direct`
  - 还带一版基于 `args/profiler_config` 的 `_should_enable_profiler_endpoints`
  - `v18-base` 已经有另一版 `_should_enable_profiler_endpoints(stage_configs)`
- 决策重点：需要统一 profiler endpoint 判断函数的签名和调用点，同时保留 request-id / arrival / finish 日志能力
- 冲突块 2：
  - `v18-base`：图片生成时直接 `request_id = f"img_gen-{random_uuid()}"`
  - `v16-base`：沿用前面已经解析出的 `request_id`
- 决策重点：如果要支持 header/body 透传 request-id，则应偏向 `v16-base` 的做法
- 冲突块 3：
  - `v18-base`：定义 `_cleanup_video`
  - `v16-base`：定义 `_run_video_generation` 包装层，负责 request-id、arrival/finish 日志和异常映射
- 决策重点：视频路径是保留“清理辅助函数”，还是引入完整的请求包装层；实际大概率应两者并存

### `vllm_omni/entrypoints/openai/serving_chat.py`

- 冲突类型：`UU`
- `v18-base`：
  - 只有在用户显式传 `num_inference_steps` 时，才覆盖默认值
- `v16-base`：
  - 从 `extra_body` 中提取 scheduler 元数据
  - 写入 `gen_params.extra_args`
  - 覆盖项包括：
    - `slo_ms`
    - `slo_target_ms`
    - `deadline_ts`
    - `estimated_cost_s`
- 决策重点：这两段逻辑并不互斥，最后很可能都该保留

## 建议的实际解冲突顺序

1. `vllm_omni/diffusion/request.py`
2. `vllm_omni/entrypoints/async_omni.py`
3. `vllm_omni/diffusion/executor/multiproc_executor.py`
4. `vllm_omni/diffusion/diffusion_engine.py`
5. `vllm_omni/diffusion/worker/diffusion_model_runner.py`
6. `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
7. `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
8. `vllm_omni/entrypoints/openai/api_server.py`
9. `vllm_omni/entrypoints/openai/serving_chat.py`
10. 最后再收 `tests`、`benchmarks`、`.pre-commit-config.yaml`

## 一句话结论

- 这次 merge 的本质不是“补几处文本冲突”，而是把 `v18-base` 的上游重构与 `v16-base` 的 scheduler / step-chunk / profiling / request-metadata 体系重新拼起来
- 真正的硬骨头在 `diffusion_engine.py`、`multiproc_executor.py`、`diffusion_model_runner.py`、`pipeline_qwen_image.py`、`pipeline_wan2_2.py`、`async_omni.py`
