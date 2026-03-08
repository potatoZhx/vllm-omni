# Qwen-Image 分辨率/步数开销测试说明

本文档对应脚本：`profile/imageGen/run_resolution_steps_bench.sh`。  
目标是测量**所有并行配置**在不同分辨率、不同推理步数下的单请求耗时，并对多次请求取平均和标准差。

## 1. 测试目标

脚本会遍历以下维度：

- 配置维度：`cases` 中的 `sp/cfg/tp/vae_use_slicing/vae_use_tiling`
- 分辨率维度：`global.resolutions`，例如 `128x128`、`512x512`、`1024x1024`
- 步数维度：`global.step_values`，默认可配 `5/10/50`
- 重复采样：每个 `(case, resolution, steps)` 组合先 warmup，再正式测量多次

每个组合的最终指标：

- `latency_mean_s`：正式测量多次请求的平均耗时（秒）
- `latency_std_s`：正式测量多次请求的标准差（秒）

## 2. 测试口径

- 服务就绪口径：`ready_path`（默认 `/v1/models`）返回 HTTP 200。
- 推理接口：`request_path`（默认 `/v1/images/generations`）。
- 请求耗时口径：使用 `curl` 的 `%{time_total}`，即完整 HTTP 请求往返总耗时。
- 首次启动耗时：`first_startup_s` 会记录，但仅用于参考，不参与每个组合 latency 的统计。

## 3. 执行流程

对每个启用的 case：

1. 启动 `vllm serve --omni`，等待 ready。
2. 遍历每个 `resolution` 和每个 `steps`。
3. 先执行 `warmup_runs_per_combo` 次 warmup 请求。
4. 再执行 `measured_runs_per_combo` 次正式请求并记录耗时。
5. 计算均值/标准差，写入 `summary.csv`。
6. case 结束后关闭服务进程，继续下一个 case。

如果某个组合请求非 200，会标记该组合为 `FAIL` 并写入原因。

## 4. 配置文件

默认配置文件：`profile/imageGen/resolution_steps_bench_cases.yaml`

### 4.1 `global` 关键字段

- `model`：模型路径
- `host`/`port`：服务监听地址
- `ready_path`：ready 检查路径
- `request_path`：图像生成请求路径
- `ready_timeout_sec`：启动后 ready 超时
- `poll_interval_sec`：ready 轮询间隔
- `stop_timeout_sec`：停服务超时
- `max_gpu_per_case`：超过该卡数的 case 会被 SKIP
- `log_root`：输出目录（相对 `profile/imageGen`）

请求参数相关：

- `prompt`、`negative_prompt`
- `num_outputs`：每次请求生成图片数（`n`）
- `warmup_runs_per_combo`
- `measured_runs_per_combo`
- `request_timeout_sec`
- `inter_run_sleep_sec`
- `seed_base`
- `use_fixed_seed`：`true` 时每次请求同种子，`false` 时递增种子
- `resolutions`：分辨率列表
- `step_values`：步数列表（如 `5, 10, 50`）
- `extra_cli_args`：透传到 `vllm serve` 的额外参数数组

### 4.2 `cases` 字段

每个 case 包含：

- `id`
- `name`
- `enabled`
- `sp`
- `cfg`
- `tp`
- `vae_use_slicing`
- `vae_use_tiling`

脚本按 `gpu = sp * cfg * tp` 计算卡数，并与 `max_gpu_per_case` 比较。

## 5. 运行方式

在工作目录 `/home/wtz2333` 下：

```bash
bash vllm-omni-wtz/profile/imageGen/run_resolution_steps_bench.sh
```

或指定配置/模型：

```bash
bash vllm-omni-wtz/profile/imageGen/run_resolution_steps_bench.sh \
  --config vllm-omni-wtz/profile/imageGen/resolution_steps_bench_cases.yaml \
  --model /path/to/model
```

## 6. 输出文件说明

默认输出根目录：`profile/imageGen/image_steps_results`

主要输出：

- `summary.csv`：每个 `(case, resolution, steps)` 一行汇总
- `case_<id>_<name>/service.log`：该 case 服务日志
- `case_<id>_<name>/runs.csv`：该 case 全部 warmup/measure 明细
- `case_<id>_<name>/combo_<resolution>_<steps>.csv`：单组合测量明细

`summary.csv` 列含义：

- `case_id, case_name`
- `gpu, sp, cfg, tp`
- `vae_use_slicing, vae_use_tiling`
- `resolution, steps`
- `first_startup_s`
- `warmup_runs, measured_runs`
- `latency_mean_s, latency_std_s`
- `status`（`PASS/FAIL/SKIP`）
- `note`（失败或跳过原因）

## 7. 结果解读建议

- 同一 case 内，比较不同 `resolution` 和 `steps` 的均值，评估算力敏感性。
- 同一 `resolution + steps` 下，比较不同 case 的均值/方差，评估并行配置收益。
- 若 `std` 较大，建议提高 `measured_runs_per_combo`（例如 10 或 20）提升稳定性。

## 8. 注意事项

- 该脚本会在每个 case 内长时间占用 GPU，请确认机器独占或低干扰。
- 高分辨率与高步数组合（如 `1536x1536 + 50`）耗时很长，建议先小规模试跑。
- 若接口返回非 200，可先查看对应 case 的 `service.log` 排查。

