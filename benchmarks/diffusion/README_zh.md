# Diffusion 在线服务 Benchmark（图像 / 视频）

该目录包含 diffusion 模型的在线服务 benchmark 脚本。
它会向 vLLM 的 OpenAI 兼容接口发送请求，并统计吞吐、时延分位数，以及可选的 SLO 达成情况。

主入口脚本：

- `benchmarks/diffusion/diffusion_benchmark_serving.py`

## 1. 快速开始

1. 启动服务：

```bash
vllm serve Qwen/Qwen-Image --omni --port 8099
```

2. 运行一个最小 benchmark：

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Qwen/Qwen-Image \
	--task t2i \
	--dataset vbench \
	--num-prompts 5
```

注意：

- benchmark 默认请求 `http://<host>:<port>/v1/chat/completions`
- 如果服务跑在其他 host 或 port，请显式传入 `--base-url`

## 2. 支持的数据集类型

通过 `--dataset` 目前支持 3 种模式：

- `vbench`：内置 prompt / 数据加载器
- `trace`：异构请求 trace，每个请求可以有不同分辨率、frames、steps
- `random`：用于快速 smoke test 的合成请求

### VBench 数据集

`vbench` 只提供 prompt 数据，以及 `i2v/i2i` 场景下的图片路径；它本身不携带每个请求的生成参数。
在这种模式下，所有请求共享 CLI 上给出的：
`--width --height --num-frames --fps --num-inference-steps`
其中 `--width` 和 `--height` 需要一起传。

示例（`t2v`）：

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
	--task t2v \
	--dataset vbench \
	--num-prompts 50 \
	--width 640 --height 480 \
	--num-frames 81 --fps 16 \
	--num-inference-steps 40
```

补充说明：

- `vbench` 也可以用于 `t2i` / `i2v` / `i2i`
- 对 `t2i`，loader 会复用 VBench 的 t2v 文本 prompt
- 对 `i2v` / `i2i`，loader 会读取 VBench 的 i2v 数据集并带上图片路径

如果你在 `i2v/i2i` 数据集场景下需要自动下载，可能需要先安装：

```bash
uv pip install gdown
```

### Trace 数据集

使用 `--dataset trace` 可以回放一份 trace 文件。trace 中可以按请求携带如下字段：

- `width`、`height`
- `num_frames`（视频）
- `num_inference_steps`
- `seed`、`fps`
- 可选 `slo_ms`（单请求 SLO 目标）

默认情况下，如果没有传 `--dataset-path`，脚本会从 HuggingFace 数据集仓库 `asukaqaqzz/Dit_Trace` 下载默认 trace。
默认文件名会依赖 `--task`，例如 `t2v` 会使用视频 trace。

当前默认值：

- `--task t2i` -> `sd3_trace.txt`
- `--task t2v` -> `cogvideox_trace.txt`

如果你要指定自己的 trace 文件，可以通过 `--dataset-path` 传入。

## 3. Benchmark 参数

### 基础参数

- `--base-url`：服务地址，脚本会调用 `.../v1/chat/completions`
- `--model`：OpenAI 兼容请求里的 `model` 字段
- `--task`：任务类型，例如 `t2i`、`t2v`、`i2i`、`i2v`
- `--dataset`：数据集模式，支持 `vbench / trace / random`
- `--num-prompts`：要发送的请求数量

常见可选参数：

- `--output-file`：把 metrics 写入 JSON 文件
- `--disable-tqdm`：关闭进度条

### 分辨率 / frames / steps：CLI 默认值与数据集字段的关系

相关参数：`--width`、`--height`、`--num-frames`、`--fps`、`--num-inference-steps`

- 对 `vbench / random`：这些 CLI 参数会作为所有请求的全局默认值
- 对 `trace`：每个请求可以自带字段，例如 `width/height/num_frames/num_inference_steps`，脚本再按下面的覆盖规则处理

`trace` 模式下的优先级规则：

- `width/height`：只要 CLI 上显式设置了 `--width` 或 `--height`，就覆盖 trace 中的请求值；否则优先使用 trace 里的值
- `num_frames`：优先使用请求级 `num_frames`，否则回退到 `--num-frames`
- `num_inference_steps`：优先使用请求级 `num_inference_steps`，否则回退到 `--num-inference-steps`

### SLO、warmup 与最大并发

通过 `--slo` 开启 SLO 评估。

- 如果 trace 里的请求已经带有 `slo_ms`，就直接使用该值
- 否则脚本会先跑 warmup 请求推断 `expected_ms`
- 如果提供了 `--warmup-request-config`，请求会复用同类型 warmup 的平均时延
- 如果没有提供 `--warmup-request-config`，脚本会回退到基于推断出的基础单位时间做线性缩放
- 最终 `slo_ms = expected_ms * --slo-scale`
- 面向 scheduler 的代价字段 `estimated_cost_s = expected_ms / 1000`

Warmup 相关参数：

- `--warmup-requests`：warmup 请求数量
- `--warmup-num-inference-steps`：当 `--warmup-request-config` 没有提供 `num_inference_steps` 时使用的兜底 steps
- `--warmup-request-config`：可选的 JSON profile 列表。warmup 请求会先基于数据集请求构造，再用这些 profile 覆盖，所以 prompt / image 输入仍与数据集一致，而分辨率、frames、steps 可以对齐到你预期的服务流量形态
- 对 `--task t2v`：warmup 请求会强制使用 `num_frames=1`，以缩短 warmup 时间并降低噪声

固定图像流量的 warmup 配置示例：

```bash
--warmup-requests 4 \
--warmup-request-config '[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]'
```

流量 / 并发相关参数：

- `--request-rate`：目标请求速率，单位 requests/second；如果设为 `inf`，脚本会立刻发送所有请求
- `--max-concurrency`：客户端允许的最大 in-flight 请求数，默认是 `1`。如果这个值过小，请求会在客户端 semaphore 后排队，实际达到的吞吐和观测到的 SLO 达成率都会被扭曲
- `--inject-scheduler-slo`：额外把 `slo_ms / slo_target_ms` 注入到请求 payload 中；只要 trace 或 warmup 推断得到了结果，`estimated_cost_s` 也会一并发送
- `--save-output-dir`：把成功生成的图像 / 视频结果保存到本地目录，便于抽样检查
