- # Global Scheduler Benchmark 中文用户指南
  
  本目录提供的是一套面向 `v18-base` global scheduler 的精简 benchmark入口。使用方式：
  
  1. 选一份 YAML 模板。
  2. 按自己的 NPU 机器修改 worker、端口、模型和并行参数。
  3. 用 `run_case.sh` 跑通 case。
  4. 可选：用 `run_suite.sh` 比较多组策略。
  5. 到 `benchmarks/diffusion/results/` 看结果。
  
  这些脚本会自动帮你做下面的事情：
  
  - 启动 global scheduler
  - 等待 worker 变成可路由且 API-ready
  - 调用 `benchmarks/diffusion/diffusion_benchmark_serving.py`
  - 保存生成后的配置、日志和 metrics
  
  
  
  ## 快速开始
  
  所有命令都在仓库根目录执行：
  
  ```bash
  cd /path/to/vllm-omni
  ```
  
  开始前，先把对应 YAML 调整成你的 NPU 环境配置。最常见的是这些字段：
  
  - `instances[*].endpoint`
  - `instances[*].launch.model`
  - `instances[*].launch.env.*VISIBLE_DEVICES`
  - `instances[*].launch.args` 里的并行参数，比如 `--usp`、`--cfg-parallel-size`
  - `benchmark.worker_ids`
  
  说明：
  
  - NPU 环境下，设备可见性建议统一使用 `ASCEND_RT_VISIBLE_DEVICES`
  - 如果你的运行时使用别的 NPU 设备环境变量，请按实际环境改 YAML
  
  
  
  ### 1. workflow：先改 YAML，再跑命令
  
  #### 1.1 选一份接近的模板
  
  - `multi_instance.qwen.yaml`
    适合 Qwen-Image 多实例对比
  - `single_instance.wan2_2.yaml`
    适合 Wan 基线
  - `multi_instance.wan2_2.yaml`
    适合 Wan 优化方案
  - `single_instance.qwen.yaml` / `single_instance.wan2_2.yaml`
    适合 smoke test
  
  #### 1.2 修改 YAML
  
  根据需要修改`policy.baseline.algorithm` 和 并行配置，即 `instances.launch`中所有`args`
  
  通常还要确认这些内容：
  
  - 端口不冲突
    - npu环境下`instances.launch.env`设置为`ASCEND_RT_VISIBLE_DEVICES`
  - 每个 worker 的设备划分不重叠
  - `benchmark.worker_ids` 和 `instances[*].id` 一一对应
  - Wan 的 `--usp`、`--cfg-parallel-size`
    和实际使用的 NPU 设备数匹配
  
  #### 1.3 在 bash 里执行命令
  
  例如，跑 Qwen 优化方案：
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml \
  CASE_NAME=short_queue_runtime_and_sjf_aging_guarded_tail \
  GLOBAL_POLICY=short_queue_runtime \
  DIFFUSION_SCHEDULER_BACKEND=step_level_request_scheduler \
  INSTANCE_POLICY=sjf_aging_guarded_tail \
  REQUEST_RATES=0.2,0.4 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  #### 1.4 查看结果
  
  结果会写到：
  
  ```bash
  benchmarks/diffusion/results/
  ```
  
  一个 case 目录里常见的文件包括：
  
  - `global_scheduler.generated.yaml`
  - `global_scheduler_server.log`
  - `instance_logs/`
  - `metrics.json` 或 `metrics_rps_<rate>.json`
  
  如果行为和你预期不一致，先看 `global_scheduler.generated.yaml`，因为它才是
  真正跑起来的最终配置。
  
  ### 2. 常用配置
  
  #### 推荐的基线和优化方案（确认yaml文件配置）
  
  下面是这份文档默认使用的几组名字和含义。
  
  - `baseline` for Qwen
    8 实例配置，使用 `multi_instance.qwen.yaml`，全局策略 `round_robin`，
    实例内策略 `fcfs`，worker 后端 `request_scheduler`
  - `baseline` for Wan
    单实例 8 卡 NPU 配置，使用 `single_instance.wan2_2.yaml`，
    其中单 worker 采用 `usp=4`、`cfg-parallel-size=2`，全局策略
    `round_robin`，实例内策略 `fcfs`，worker 后端 `request_scheduler`
  - `short_queue_runtime_and_sjf_aging_guarded_tail`
    这是优化方案名字，表示全局策略 `short_queue_runtime`，worker 使用
    `step_level_request_scheduler`，实例内策略
    `sjf_aging_guarded_tail`
  - optimized for Qwen
    8 实例配置，使用 `multi_instance.qwen.yaml`，全局策略
    `short_queue_runtime`，实例内策略 `sjf_aging_guarded_tail`
  - optimized for Wan
    双实例 NPU 配置，使用 `multi_instance.wan2_2.yaml`，每个实例使用
    `usp=2`、`cfg-parallel-size=2`，全局策略 `short_queue_runtime`，
    实例内策略 `sjf_aging_guarded_tail`
  
  #### Qwen 基线
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml \
  CASE_NAME=baseline \
  GLOBAL_POLICY=round_robin \
  DIFFUSION_SCHEDULER_BACKEND=request_scheduler \
  ENABLE_STEP_CHUNK=0 \
  REQUEST_RATES=0.2,0.4 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  #### Qwen 优化方案
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml \
  CASE_NAME=short_queue_runtime_and_sjf_aging_guarded_tail \
  GLOBAL_POLICY=short_queue_runtime \
  DIFFUSION_SCHEDULER_BACKEND=step_level_request_scheduler \
  INSTANCE_POLICY=sjf_aging_guarded_tail \
  REQUEST_RATES=0.2,0.4 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  #### Wan 基线
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml \
  CASE_NAME=baseline \
  GLOBAL_POLICY=round_robin \
  DIFFUSION_SCHEDULER_BACKEND=request_scheduler \
  ENABLE_STEP_CHUNK=0 \
  REQUEST_RATES=0.05,0.1 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  - 单 worker
  - 8 张设备卡
  - `--usp 4`
  - `--cfg-parallel-size 2`
  
  #### Wan 优化方案
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.wan2_2.yaml \
  CASE_NAME=short_queue_runtime_and_sjf_aging_guarded_tail \
  GLOBAL_POLICY=short_queue_runtime \
  DIFFUSION_SCHEDULER_BACKEND=step_level_request_scheduler \
  INSTANCE_POLICY=sjf_aging_guarded_tail \
  REQUEST_RATES=0.05,0.1 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  - 两个 worker
  - 每个 worker 4 张设备卡
  - 每个 worker `--usp 2`
  - 每个 worker `--cfg-parallel-size 2`
  
  ### 3. Benchmark 模式
  
  通过 `BENCHMARK_MODE` 支持两种模式：
  
  - `fixed_duration`
  - `fixed_num_prompts`
  
  `fixed_duration` 的语义：
  
  - 使用 `REQUEST_RATES` 和 `NUM_PROMPTS_DURATION_SECONDS`
  - 计算 `num_prompts = ceil(request_rate * duration_seconds)`
  - 它不是 benchmark 内部“到点硬停止”的 wall-clock 截止
  - 更准确地说，这是“按目标时长推导请求总数”
  
  `fixed_num_prompts` 的语义：
  
  - 使用 `REQUEST_RATES` 和 `FIXED_NUM_PROMPTS`
  - 每个 request rate 都发送相同数量的请求
  
  示例：
  
  ```bash
  BENCHMARK_MODE=fixed_duration \
  NUM_PROMPTS_DURATION_SECONDS=600 \
  REQUEST_RATES=0.2,0.4 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  ```bash
  BENCHMARK_MODE=fixed_num_prompts \
  FIXED_NUM_PROMPTS=20 \
  REQUEST_RATES=0.2,0.4 \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
  ```
  
  
  
  ### 4. 用 `run_suite.sh` 批量比较
  
  当你已经确认单 case 能跑通后，可以用 `run_suite.sh` 跑多组算法对比。
  
  Qwen 对比示例：
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.qwen.yaml \
  REQUEST_RATES=0.2,0.4 \
  CASE_MATRIX=$'baseline|round_robin|fcfs|0\nshort_queue_runtime_and_sjf_aging_guarded_tail|short_queue_runtime|sjf_aging_guarded_tail|1' \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
  ```
  
  Wan 优化方案示例：
  
  ```bash
  BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/multi_instance.wan2_2.yaml \
  REQUEST_RATES=0.05,0.1 \
  CASE_MATRIX=$'short_queue_runtime_and_sjf_aging_guarded_tail|short_queue_runtime|sjf_aging_guarded_tail|1' \
  ./benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
  ```
  
  如果你希望把 Wan 基线也放进报告里，建议单独再跑一个 suite。因为：
  
  - Wan 基线用的是 `single_instance.wan2_2.yaml`
  - Wan 优化方案用的是 `multi_instance.wan2_2.yaml`
  
  这两者不是同一个模板。
  
  ## 执行流程
  
  每个 case 的执行过程如下：
  
  1. 读取基础 YAML 配置
  2. 套用环境变量覆盖
  3. 重写 worker launch args，确保选中的 diffusion scheduler backend 被一致应用
  4. 在输出目录中写出 generated config
  5. 启动 `python -m vllm_omni.global_scheduler.server --config <generated-config>`
  6. 等待 `/health`、`/instances` 和 worker `/v1/models` 就绪
  7. 调用 `diffusion_benchmark_serving.py` 对 scheduler URL 发压
  8. 关闭 scheduler，并保留所有产物
  
  `run_suite.sh` 会对 `CASE_MATRIX` 中的每一行重复这个流程，最后额外生成：
  
  - `summary.json`
  - `summary.csv`
  
  ## 配置说明
  
  这一节是完整参考，说明这个目录里所有实际可用的配置入口。
  
  ### A. `run_case.sh` 支持的环境变量
  
  - `BASE_CONFIG`
    基础 YAML 路径。默认是 `single_instance.qwen.yaml`
  - `GLOBAL_POLICY`
    覆盖 `policy.baseline.algorithm`
    支持：`min_queue_length`、`round_robin`、`short_queue_runtime`
  - `INSTANCE_POLICY`
    覆盖 worker `--instance-scheduler-policy`
    只在 `DIFFUSION_SCHEDULER_BACKEND=step_level_request_scheduler`
    时有意义
    支持：`fcfs`、`sjf`、`sjf_aging`、`sjf_aging_guarded`、
    `sjf_aging_guarded_tail`、`p95-first`
  - `DIFFUSION_SCHEDULER_BACKEND`
    覆盖 worker `--diffusion-scheduler-backend`
    支持：`request_scheduler`、`step_level_request_scheduler`
  - `ENABLE_STEP_CHUNK`
    显式控制 `--diffusion-enable-step-chunk`
    可用值：空、`0`、`1`
    `step_level_request_scheduler` 必须启用 step chunk
    `request_scheduler` 与 `ENABLE_STEP_CHUNK=1` 不兼容
  - `REQUEST_RATES`
    逗号或空格分隔的 request rate，例如 `0.2,0.4,0.6`
  - `REQUEST_RATE`
    兼容旧用法的别名，只有在没设置 `REQUEST_RATES` 时才使用
  - `BENCHMARK_MODE`
    `fixed_duration` 或 `fixed_num_prompts`
  - `NUM_PROMPTS_DURATION_SECONDS`
    `fixed_duration` 使用
  - `REQUEST_DURATION_S`
    `NUM_PROMPTS_DURATION_SECONDS` 的旧别名
  - `FIXED_NUM_PROMPTS`
    `fixed_num_prompts` 使用
  - `CASE_NAME`
    可选，显式指定 case 名
  - `RUN_TAG`
    输出目录后缀，默认是时间戳
  - `OUT_DIR`
    显式指定 case 输出目录
  - `BENCH_OUTPUT_FILE`
    显式指定 benchmark metrics 路径
  - `SCHEDULER_LOG_FILE`
    显式指定 scheduler 日志路径
  - `WORKER_IDS`
    只跑部分 worker，逗号或空格分隔
  - `BENCHMARK_MODEL`
    覆盖 `benchmark.model`
  - `BENCHMARK_BACKEND`
    覆盖 `benchmark.backend`
  - `BENCHMARK_TASK`
    覆盖 `benchmark.task`
  - `BENCHMARK_DATASET`
    覆盖 `benchmark.dataset`
    benchmark 脚本支持：`random`、`trace`、`vbench`
  - `BENCHMARK_DATASET_PATH`
    覆盖 `benchmark.dataset_path`
  - `BENCHMARK_RANDOM_REQUEST_CONFIG`
    覆盖 `benchmark.random_request_config`
  - `BENCHMARK_MAX_CONCURRENCY`
    覆盖 `benchmark.max_concurrency`
  - `BENCHMARK_WARMUP_REQUESTS`
    覆盖 `benchmark.warmup_requests`
  - `BENCHMARK_WARMUP_NUM_INFERENCE_STEPS`
    覆盖 `benchmark.warmup_num_inference_steps`
  
  注意：
  
  - 如果你要改 warmup profile，请直接改 YAML 里的
    `benchmark.warmup_request_config`
  
  ### Warmup 语义
  
  `benchmark.warmup_request_config` 会透传为：
  
  - `diffusion_benchmark_serving.py --warmup-request-config`
  
  它的行为是：
  
  - warmup 请求仍然基于主 benchmark 选择的数据集构造
  - 然后再用 profile 覆盖 `width`、`height`、`num_frames`、`fps`、
    `num_inference_steps` 等字段
  - 如果你希望 warmup 覆盖一组固定服务形态，这是推荐做法
  
  当前限制：
  
  - 没有单独的 `warmup_dataset`
  - warmup 不能使用与主 benchmark 不同的数据集
  
  ### B. `run_suite.sh` 额外支持的环境变量
  
  `run_suite.sh` 支持上面所有变量，另外还支持：
  
  - `SUITE_NAME`
    suite 输出目录名
  - `OUT_ROOT`
    显式指定 suite 输出根目录
  - `CASE_MATRIX`
    每行一个 case
  
  支持的行格式：
  
  - `case_name|global_policy`
  - `case_name|global_policy|instance_policy`
  - `case_name|global_policy|instance_policy|scheduler_backend_flag`
  
  第 4 列 `scheduler_backend_flag` 支持：
  
  - `0`：使用 `request_scheduler`
  - `1`：使用 `step_level_request_scheduler`
  - 也可以直接写 backend 名
  
  补充语义：
  
  - 当第 4 列为 `0` 时，orchestrator 会按 `request_scheduler` 运行
  - 这时不会应用 `--diffusion-enable-step-chunk`
  - 这时 `instance_policy` 列和 `INSTANCE_POLICY` 环境变量都会被忽略
  
  示例：
  
  ```bash
  CASE_MATRIX=$'baseline|round_robin|fcfs|0\nshort_queue_runtime_and_sjf_aging_guarded_tail|short_queue_runtime|sjf_aging_guarded_tail|1'
  ```
  
  ```bash
  CASE_MATRIX=$'mql|min_queue_length\nrr|round_robin\nsqr|short_queue_runtime'
  ```
  
  ```bash
  CASE_MATRIX=$'fcfs|round_robin|fcfs|1\nsjf|round_robin|sjf|1\nsjf_aging_guarded_tail|round_robin|sjf_aging_guarded_tail|1'
  ```
  
  ### C. YAML 结构
  
  基础 YAML 顶层有 5 个部分：
  
  - `server`
  - `scheduler`
  - `policy`
  - `benchmark`
  - `instances`
  
  #### `server`
  
  - `host`
    scheduler 监听地址。即使写 `0.0.0.0`，benchmark 侧也会按需改用
    `127.0.0.1`
  - `port`
    scheduler 端口
  - `request_timeout_s`
    scheduler 转发请求的超时时间
  - `instance_health_check_interval_s`
    worker 健康检查间隔
  - `instance_health_check_timeout_s`
    单次健康检查超时时间
  - `instance_health_check_failures_before_unhealthy`
    连续失败多少次后标记 unhealthy
    scheduler 支持这个字段，即使模板里没写
  
  #### `scheduler`
  
  - `tie_breaker`
    `random` 或 `lexical`
    当策略打分相同，用它来决策
  - `ewma_alpha`
    EWMA 平滑系数，必须在 `(0, 1]`
  
  #### `policy.baseline`
  
  - `algorithm`
    全局路由策略
    支持：`min_queue_length`、`round_robin`、`short_queue_runtime`
  - `runtime_profile_path`
    可选，给全局 runtime estimator 用的 JSON runtime profile
    对 `short_queue_runtime` 特别有用
  
  这些全局策略的含义：
  
  - `min_queue_length`
    在这轮迁移里，实际效果接近“选当前 inflight 最少的 worker”
  - `round_robin`
    按顺序轮转，不看请求代价
  - `short_queue_runtime`
    选择预计 outstanding runtime 最小的 worker
    当请求带有 `estimated_cost_s` 时效果最好
  
  #### `benchmark`
  
  - `worker_ids`
    本次 benchmark 纳入哪些 worker
  - `worker_ready_timeout_s`
    等待 worker 变成可路由且 API-ready 的超时时间
  - `model`
    传给 `diffusion_benchmark_serving.py` 的模型名
  - `backend`
    benchmark 命中的后端族
    常见值：
    Qwen 用 `vllm-omni`
    Wan2.2 用 `v1/videos`
  - `task`
    例如 `t2i`、`t2v`
  - `dataset`
    `random`、`trace` 或 `vbench`
  - `dataset_path`
    可选，本地 trace 或数据文件路径
  - `random_request_config`
    随机请求混合分布
  - `warmup_requests`
    warmup 请求数
  - `warmup_num_inference_steps`
    warmup fallback step 数
  - `warmup_request_config`
    warmup profile JSON
  - `max_concurrency`
    benchmark 客户端侧 in-flight 上限
  - `output_file`
    通常由 orchestrator 自动管理，不建议手工维护
  
  实践建议：
  
  - 先用 `random` 做 smoke test 和合成负载测试
  - 复现真实异构流量时用 `trace`
  - 想用内置 prompt 时用 `vbench`
  
  #### `instances[]`
  
  每个 worker 条目可以包含：
  
  - `id`
    逻辑 worker id
  - `endpoint`
    worker base URL，例如 `http://127.0.0.1:9001`
    不要带 path
  - `instance_type`
    runtime 估算和 runtime profile 会用到
  - `numa_node`
    可选。如果系统里有 `numactl`，启动时会自动加前缀
  - `backends`
    该实例允许被哪些 backend 路由到
  - `launch`
    如何启动 worker
  - `stop`
    如何停止 worker
  
  `launch` 里常见字段：
  
  - `executable`
    通常是 `vllm`
  - `model`
    `vllm serve` 使用的模型名
  - `args`
    追加到 `vllm serve <model> --port <endpoint_port>` 后面的参数
  - `env`
    worker 进程环境变量
  
  `stop` 里支持的占位符：
  
  - `{instance_id}`
  - `{endpoint}`
  - `{endpoint_host}`
  - `{endpoint_port}`
  
  ### D. worker 启动参数改写规则
  
  orchestrator 会主动规范化这些 worker 参数：
  
  - `--diffusion-scheduler-backend`
  - `--diffusion-enable-step-chunk`
  - `--instance-scheduler-policy`
  
  规则如下：
  
  - 如果 `DIFFUSION_SCHEDULER_BACKEND=request_scheduler`
    会移除实例内策略
  - 如果 `DIFFUSION_SCHEDULER_BACKEND=step_level_request_scheduler`
    会强制要求 step chunk
  - 如果设置了 `INSTANCE_POLICY`
    它会覆盖 YAML 中已有的实例内策略
  - 如果没设置 `INSTANCE_POLICY`
    就沿用 YAML 里原有的策略
  
  
  
  ## 目录文件说明
  
  - `run_case.sh`
    单 case 入口，本质上是 `python3 orchestrate.py case`
  
  - `run_suite.sh`
    多 case 入口，本质上是 `python3 orchestrate.py suite`
  
  - `orchestrate.py`
    主实现。负责读取环境变量、重写配置、启动 scheduler、等待就绪、
    调 benchmark、保存结果
  
  - `single_instance.qwen.yaml`
    Qwen 单实例模板，适合 smoke test
  
  - `multi_instance.qwen.yaml`
    Qwen 8 实例模板，适合做全局调度对比
  
  - `single_instance.wan2_2.yaml`
    Wan 单实例模板。默认是 8 卡 NPU 风格配置，`usp=4`、
    `cfg-parallel-size=2`
  
  - `multi_instance.wan2_2.yaml`
    Wan 双实例模板。默认每个实例 4 卡，`usp=2`、
    `cfg-parallel-size=2`
  
    
  
  ## 输出文件说明
  
  每个 case 目录里通常会有：
  
  - `global_scheduler.generated.yaml`
    环境变量覆盖和参数归一化后的最终配置
  - `global_scheduler_server.log`
    scheduler 的 stdout / stderr
  - `instance_logs/`
    如果 scheduler 把 worker 日志写到这里，就会出现在该目录
  - `metrics.json` 或 `metrics_rps_<rate>.json`
    benchmark 结果
  
  suite 根目录还会额外包含：
  
  - `summary.json`
  - `summary.csv`
  
  `summary.csv` 当前包含这些列：
  
  - `case`
  - `request_rate`
  - `completed`
  - `throughput_qps`
  - `latency_p50`
  - `latency_p95`
  - `latency_p99`
  - `backend`
  - `model`
  - `metrics_file`
  
  ## 其他有用的信息
  
  - `short_queue_runtime` 想发挥作用，请尽量让请求带上 `estimated_cost_s`
    benchmark 脚本可以通过 warmup 或 trace 推断并注入这个字段
  - `request_scheduler`
    是原始 request-level worker 执行路径
  - `step_level_request_scheduler`
    是 step-level worker 路径
    `sjf`、`sjf_aging`、`sjf_aging_guarded`、
    `sjf_aging_guarded_tail`、`p95-first`
    都依赖这个后端
  - 当前 orchestrator 的支持范围：
    全局策略 `min_queue_length`、`round_robin`、`short_queue_runtime`；
    worker backend 归一化；
    `benchmark.warmup_request_config`；
    单 case 和 suite 两种运行方式
  - 当前明确不做：
    scheduler 侧等待或 admission blocking、
    `diffusion_engine_max_concurrency`、
    orchestration 侧自动合成 guarded / tail / p95 专属调参参数
  - 单 worker 场景下，全局策略差异通常不明显
    真正做全局调度对比，优先使用 `multi_instance.*.yaml`
  - `fixed_duration` 不是 benchmark 内部硬性 wall-clock 截止
    只是按目标时长推导请求数
  - 模板里虽然有 `benchmark.auto_stop: true`
    但 `orchestrate.py` 当前不会使用它
  
  
