# VideoGen Parallel Profiling

该目录用于对 Wan2.2 T2V 在不同卡数/并行配置下进行离线推理延迟统计。

## 固定模型

默认模型路径已固定为：

`/data2/group_谈海生/mumura/models/Wan2.2-T2V-A14B-Diffusers`

可通过 `--model` 覆盖。

## 覆盖范围

- 请求类型：按 traits 扩展为 24 种（2 分辨率 × 4 帧数 × 3 steps）
  - 配置文件：`request_types_24.json`
- 并行配置：按 1/2/4/8 卡矩阵
  - 配置文件：`parallel_matrix.json`
  - 所有配置显式设置 `vae_patch_parallel_size=卡数`
  - 默认启用非 HSDP 方案；HSDP 配置保留在矩阵中，默认 `enabled=false`。

## 运行

流程说明：

- 对每个并行配置，仅启动一次 worker 并加载一次模型
- 先执行 warmup（`--warmup-iters`）
- 再执行 24 请求类型 × `--repeats` 次推理，记录离线延迟
- 每条请求执行后立即增量落盘到 `worker_results.json`，中断后可续跑
- 默认开启 `--resume`，会跳过已成功（status=ok）的请求项

在仓库根目录执行：

```bash
python profile/videoGen/run_video_parallel_bench.py \
  --card-counts 1,2,4,8 \
  --gpu-device-ids 0,1,2,3,4,5,6,7 \
  --warmup-iters 1 \
  --repeats 3
```

可选容错参数：

- `--no-resume`：关闭断点续跑
- `--request-fail-fast`：某个请求失败就让 worker 立即退出
- `--fail-fast`：某个并行配置失败就终止整个主流程
- `--request-timeout-seconds <N>`：限制 worker 内普通请求最长时间
- `--warmup-timeout-seconds <N>`：单独限制 warmup 最长时间（建议设置得更大，或保持 0 不限制）
- `--timeout-grace-seconds <N>`：请求超时后先发 SIGTERM，等待 N 秒，再对残留进程发 SIGKILL
- `--worker-timeout-seconds <N>`：限制每个并行配置 worker 的最长运行时间；超时会强制杀掉该配置进程组并进入下一配置

例如（推荐）：

```bash
python profile/videoGen/run_video_parallel_bench.py \
  --card-counts 1,2,4,8 \
  --gpu-device-ids 0,1,2,3,4,5,6,7 \
  --warmup-iters 1 \
  --repeats 3 \
  --request-timeout-seconds 300 \
  --warmup-timeout-seconds 1200 \
  --timeout-grace-seconds 20 \
  --worker-timeout-seconds 3600
```

先做计划校验（不实际执行）：

```bash
python profile/videoGen/run_video_parallel_bench.py --dry-run
```

## 输出

结果输出目录：

`profile/videoGen/results/<timestamp>/`

其中：

- `summary_runs.csv`：每次运行（配置 × 请求类型 × repeat）的明细结果
- `plan.json`：本次实验计划快照
- `<config_dir>/worker.log`：该并行配置下 worker 日志
- `<config_dir>/worker_results.json`：该并行配置下全部请求结果（含 warmup 参数与聚合）

## 汇总

```bash
python profile/videoGen/aggregate_video_parallel_results.py \
  --summary-csv profile/videoGen/results/<timestamp>/summary_runs.csv
```

输出：`summary_agg.csv`（按配置+请求类型聚合）。
