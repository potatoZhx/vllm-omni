# Step-chunk 抢占调度相对 main 的代码审查与延迟异常分析

> 说明：当前仓库没有本地 `main` 分支引用，且无法访问远程仓库；本报告以 `d935e78` 的第二父提交（merge 时的 main 侧快照）作为对比基线。

## 1. 关键改动对比结论

1. 执行器默认调度器从 `Scheduler` 切换到 `Stage1Scheduler`，调度路径完全重构（新增排队、策略选择、抢占回队）。
2. `DiffusionEngine.step()` 改为 chunk 循环驱动：当 `diffusion_enable_step_chunk=True` 时会持续调度同一请求，直至 `output.finished=True`。
3. `DiffusionModelRunner.execute_model()` 在 step-chunk 模式下新增上下文保存/恢复路径（`prepare_generation/step_generation/finalize_generation`），并通过 `executed_steps` 回写进度。
4. Wan2.2 / Qwen-Image pipeline 新增 scheduler state 捕获与恢复，支持 chunk 续跑。
5. 配置层增加隐式行为：某些策略（如 `p95-first`、`sjf_aging_guarded`）会强制打开 `diffusion_enable_step_chunk` 与 `diffusion_enable_chunk_preemption`。

## 2. 实现正确性评估（结论：主体方向正确，但存在“性能比较不等价”与潜在性能偏差点）

### 2.1 正确性主链路

- 从引擎层看，step-chunk 不会提前返回最终结果：`DiffusionEngine.step()` 会循环直到 `finished=True`，因此“提前返回导致看起来更快”这类逻辑错误在主链路上基本被排除。
- 从模型层看，Wan2.2 的 chunk 执行会保存并恢复 scheduler 内部状态（包括 `_step_index`、`model_outputs` 等），这是保证多次 dispatch 结果一致性的必要条件。

### 2.2 需要重点警惕的问题

1. **缓存刷新与 chunk 进度耦合不完整（高可疑）**
   - `execute_model()` 每次 dispatch 都会调用 `cache_backend.refresh(..., num_inference_steps=req.sampling_params.num_inference_steps)`。
   - 但 step-chunk 的真实进度来自 `ctx.current_step`，并未把“已执行步数”传给缓存后端。
   - 若缓存后端（特别是 TeaCache / CacheDiT）内部使用“从第 0 步开始”的上下文，这会在每个 chunk 重置缓存状态，导致**缓存命中模式与主干一次性推理不同**，可能出现“异常变快但质量/轨迹不等价”。

2. **调度策略会隐式开启 step-chunk + 抢占（配置语义变化）**
   - 即使你“看起来是同一份配置文件”，只要策略名变了，运行路径可能已变化。
   - 这会直接破坏与 main 的 A/B 等价性（main 可能仍是单次 run-to-completion 路径）。

3. **调度器层与基线不同（不可忽略）**
   - 执行器中默认 scheduler 已替换为 `Stage1Scheduler`。即使单请求场景，也会经历新的排队/状态更新/metrics 注入逻辑，已经不是 main 的同路径比较。

## 3. 为什么会“单条请求反而更快”——最可能原因排序

1. **A/B 不等价（最高概率）**
   - 新分支可能因为策略映射，自动走了 step-chunk + preemption；main 走的是单次 forward。

2. **缓存行为被 chunk 化重置（高概率）**
   - 每个 chunk 刷新 cache context，可能让后端在“每段前几步”使用更激进/不同的缓存策略，减少有效计算量，从而显著降时。

3. **基线统计口径不同（中概率）**
   - 新版输出 metrics 中 `scheduler_execute_ms`/`queue_wait_ms` 是按 chunk 聚合的；若外部只取部分指标，可能造成“看起来更快”。

4. **模型求解轨迹发生漂移（中概率）**
   - 虽然已新增 scheduler state capture/restore，但如果某些 pipeline / backend 状态未纳入 context（例如缓存后端、个别运行态），也可能导致轨迹变化，进而影响耗时。

## 4. 建议的快速验证（可直接执行）

1. 固定同一模型、同一 seed、同一 prompt，分别测试：
   - A: `diffusion_enable_step_chunk=false`
   - B: `diffusion_enable_step_chunk=true`, `diffusion_enable_chunk_preemption=false`
   - C: `diffusion_enable_step_chunk=true`, `diffusion_enable_chunk_preemption=true`
2. 三组都关闭 cache backend（`cache_backend=none`），先验证“纯算子耗时”是否仍异常下降。
3. 若 B/C 显著快于 A，再打开 cache backend 复测；若差异突然变大，基本可定位为缓存与 chunk 进度耦合问题。
4. 对同一请求比对最终输出质量（图像/视频 hash、CLIP score 或帧差），确认“快”是否伴随结果漂移。

## 5. 结论

- 从代码实现看，step-chunk 抢占主流程（引擎循环 + pipeline 状态恢复）总体是成立的；
- 但你们观测到“单条请求显著变快”并非不可能，最可疑根因是：**配置语义变化导致路径不等价 + cache 刷新未与 chunk 已执行进度对齐**。
- 建议优先按第 4 节做三组 A/B/C 与 cache 开关实验，通常能在 1~2 轮内把问题钉住。


## 6. 针对最新反馈的补充澄清

1. **关于 FCFS 是否会隐式开启抢占**
   - 你们的判断是对的：`fcfs` 不在“自动开启 step-chunk + preemption”的策略列表中。
   - 代码里只有 `{p95-first, p95-first-deadline, p95-bucket-sjf, p95-bucket-sjf-normalized, slack_hybrid, sjf_aging_guarded}` 会在配置校验阶段强制打开两项开关。
   - 因此在严格 `fcfs` 且未显式打开相关开关时，不应把“隐式开启抢占”作为首要原因。

2. **缓存刷新与 chunk 进度耦合：是正确性风险，不是确定性的优化**
   - `execute_model()` 在每次 dispatch 都会 `cache_backend.refresh(..., num_inference_steps=总步数)`，但没有把 `current_step` 进度交给 cache backend。
   - 若 backend 基于“步号/历史状态”决定跳算或复用，这种“每 chunk 刷新但不告知已执行步数”的行为会让缓存决策与一次性完整推理不一致。
   - 结果可能有三种：
     1) 变快且结果偏移；
     2) 变慢；
     3) 波动增大。
   - 所以它更像是**潜在语义不一致/正确性风险**，而不是可依赖的性能优化。

3. **你们用的是 `duration_s` 口径**
   - 这意味着“只看 chunk 内部某个局部指标”导致的统计口径误差基本可以排除，分析应聚焦真实执行路径和算子量变化。

4. **启动命令未显式传 `cache_backend` 的补充点**
   - 配置构造会读取环境变量 `DIFFUSION_CACHE_BACKEND` / `DIFFUSION_CACHE_ADAPTER` 作为回退；若环境里有值，仍可能启用缓存后端。
   - 若环境变量也没有，则默认 `cache_backend=none`。

5. **在“FCFS + 无缓存 + duration_s”前提下，更该优先排查的点**
   - 预热路径是否一致（是否都完成 dummy run 后再压测）。
   - 请求参数是否完全一致（尤其 `num_inference_steps` / 分辨率 / frame 数 / guidance）。
   - pipeline 改造后是否存在“实际执行步数 < 配置步数”的情况（重点核对日志中的 `executed_steps` 与 `total_steps`）。
   - 同 seed 下结果是否一致（若快很多但输出差异明显，通常意味着有效计算量或轨迹已变）。

## 7. 推理调用路径（Qwen-Image / Wan）与 dispatch、cache 刷新语义

### 7.1 端到端调用链路（服务侧到模型侧）

1. `DiffusionEngine.step(request)` 是入口，内部调用 `self.add_req_and_wait_for_response(request)`。
2. `MultiprocDiffusionExecutor.add_req()` 转发给 `Stage1Scheduler.add_req()`。
3. `Stage1Scheduler.add_req()` 把请求入队、选中后通过 `mq.enqueue({type: "rpc", method: "generate", args: (request,)})` 发给 worker。
4. worker 收到 RPC 后执行 `execute_method("generate", request)`，最终调用 `DiffusionWorker.generate()` → `DiffusionWorker.execute_model()` → `DiffusionModelRunner.execute_model(req)`。
5. rank0 把 `DiffusionOutput` 回传 scheduler，scheduler 再回给 engine。

### 7.2 什么是 dispatch

- 在这里 **dispatch = scheduler 把一个 request（或其一个 chunk）下发给 worker 执行一次 `generate` RPC**。
- 不抢占时：通常 1 个请求对应 1 次 dispatch（完成即返回 finished=True）。
- 抢占时：同一请求会多次 dispatch（每次做一段 steps）。

### 7.3 Qwen-Image / Wan 在 `execute_model` 内到底怎么跑

`DiffusionModelRunner.execute_model()` 分两条路径：

- `diffusion_enable_step_chunk=False`：直接 `pipeline.forward(req)`。
- `diffusion_enable_step_chunk=True`：
  1) 取/建 `ctx`（`prepare_generation`）
  2) 按 `steps_this_turn` 执行 `step_generation`
  3) 若 finished 则 `finalize_generation`，否则返回 unfinished 的 `DiffusionOutput`

而 Qwen/Wan 的 `forward()` 目前都兼容了这两条：
- 在 step-chunk 关闭时，仍走“prepare + 一次性跑完 + finalize”的 run-to-completion 语义。

### 7.4 `cache_backend.refresh` 的真实含义

- `refresh` 是“每次执行前同步/重置 cache backend 当前上下文”，不是“强制启用 cache”。
- 代码有保护条件：
  - `self.cache_backend is not None`
  - `self.cache_backend.is_enabled()`
- 所以在你确认 `cache_backend=none` 且环境变量也无覆盖时，`self.cache_backend` 会是 `None`，该分支不会执行。

### 7.5 你当前场景下（无缓存、无抢占）它会不会导致波动？

- 结论：**基本不会**。因为 refresh 分支根本不进。
- 这意味着你观测到的速度波动/变快，优先看：
  1) 实际是否进入 step-chunk 路径（即 `diffusion_enable_step_chunk` 最终值）；
  2) 相同 benchmark 下，是否存在运行态差异（并行度、编译状态、负载干扰）；
  3) 是否真的执行了相同步数（看 `executed_steps` 与 `total_steps` 日志）。
