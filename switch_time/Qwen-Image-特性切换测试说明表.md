# Qwen/Qwen-Image 特性切换测试说明（最多 8 卡）

**Ready 口径**：本测试以**控制面 ready** 为准，即 `/v1/models` 返回 200；不要求 first image ready（不发送推理请求）。需知：`/v1/models` 200 仅表示 HTTP API 已拉起，不保证 diffusion stage 一定健康（若 stage 在 warm-up 报错，orchestrator 在 600s 超时后仍会启动 API 并返回 200，此时实际无法推理；若需严格校验可另行发一次最小图生图请求）。

## 1. 测试背景与目标

当前业务是 `Qwen/Qwen-Image` 文生图（T2I）在线服务。在实际运行中，可能需要根据流量、显存、延迟目标动态切换配置特性（例如 `TP`、`SP`、`FP8`、`TeaCache` 等），这个测试并不实际处理请求。

本测试的核心目标不是只看“单一配置下的峰值性能”，而是评估：

1. **从旧特性配置切换到新特性配置的可用性恢复时间**；
2. **不同特性在 1~8 卡范围内的综合性价比**（不要求一定用满 8 卡）。

---

## 2. 术语纠正与统一口径

为避免歧义，建议把“杀旧进程、起新进程并计时”统一描述为：

- **配置切换总耗时（Switch Time）**  
  从发出旧服务下线指令开始，到新服务达到 `Ready`（健康检查通过，可接受真实请求）为止。

并拆成两个子指标：

- **旧服务下线耗时（Stop Time）**  
  从执行 `kill/stop` 开始，到旧进程完全退出。
- **新服务上线耗时（Startup-to-Ready Time）**  
  从执行启动命令开始，到 `/v1/models` 返回 200（健康检查通过）。

公式：

`Switch Time = Stop Time + Startup-to-Ready Time`

---

## 3. 测试范围与特性组合规则

模型固定为：`Qwen/Qwen-Image`。硬件：单机最多 8 卡（可 1/2/4/8 卡）。

### 3.1 特性互斥与可组合关系（依据项目代码）

以下根据 vllm-omni 中 `serve.py`、`registry.py`、`diffusion_model_runner.py` 等整理：

| 特性 | 取值 / 说明 | 互斥或约束 |
|------|-------------|------------|
| **Cache** | `none` \| `tea_cache` \| `cache_dit` | **三选一**（`--cache-backend` 仅能取其一） |
| **SP（Ulysses/Ring）** | 1（关）或 2/4/8（开，如 `--ulysses-degree 2`） | 可与 TP、CFG 组合；与 TP、CFG 共同满足约束见下 |
| **CFG-Parallel** | 1 或 2（`--cfg-parallel-size`） | 可与 TP、SP 组合 |
| **Tensor-Parallel（TP）** | 1 / 2 / 4 / 8 | 可与 SP、CFG 组合 |
| **CPU Offload（Layerwise）** | 关 / 开（`--enable-cpu-offload`、`--enable-layerwise-offload`） | 可与其他特性自由组合 |
| **FP8 Quantization** | 关 / 开 | **与其它量化互斥**（同一进程仅一种量化） |
| **VAE-Patch-Parallel** | — | **Qwen-Image 不支持**（仅 SD3 / Z-Image / NextStep11 等在 allowlist 中），待测表不包含此项 |

**并行约束（单进程占用 GPU 数）**：

- 必须满足：`TP × SP × CFG ≤ 8`（且通常等于为该 stage 分配的 GPU 数）。
- 例如：TP=4、SP=2、CFG=1 ⇒ 4×2×1=8，合法；TP=4、SP=2、CFG=2 ⇒ 16>8，非法。

据此，**Qwen-Image 待测特性**仅包含：Cache（三选一）、SP、CFG-Parallel、TP、CPU Offload、FP8；不包含 VAE-Patch-Parallel。

---

## 4. 测试方法（推荐）

每个“特性组合 + 卡数”作为一个测试点，按以下流程执行：

**首次启动（单独记录，不参与后续统计）**

- 环境就绪后，**先做一次「首次启动」**：从执行启动命令到 `/v1/models` 返回 200，记下该时间，填入下表「首次 Startup(s)」列。该次可能因冷启动（CUDA/内核/模型首次加载）明显更慢，仅作参考，**不参与 mean/std 计算**。

**正式 20 次样本（用于 Stop/Startup/Switch 的 mean/std）**

1. 在首次启动完成后，开始计时，执行旧配置下线（Stop）；
2. 立即启动新配置服务，轮询 `/v1/models` 直到 200，得到本次的 `Stop Time`、`Startup-to-Ready Time`、`Switch Time`；
3. 重复“下线 → 启动”上述流程，**共 20 次**（即第 2～21 次启动），得到 20 个样本；
4. 将 20 个样本写入脚本日志或独立 CSV，便于复盘；
5. 基于这 20 个样本计算均值和标准差，填入下表 Stop/Startup/Switch 的 mean/std 列, 或者和特性组合一起，规整地打印到日志

---

## 5. 待测组合与测试记录表

### 5.1 合法并行配置（TP × SP × CFG ≤ 8）

以下 (TP, SP, CFG) 满足单进程占用的 GPU 数 ≤ 8，可与 Cache/CPU Offload/FP8 任意合法组合：

| (TP, SP, CFG) | GPU 数 | (TP, SP, CFG) | GPU 数 |
|---------------|--------|---------------|--------|
| (1,1,1) | 1 | (2,2,2) | 8 |
| (1,1,2) | 2 | (2,4,1) | 8 |
| (1,2,1) | 2 | (4,1,1) | 4 |
| (1,2,2) | 4 | (4,1,2) | 8 |
| (1,4,1) | 4 | (4,2,1) | 8 |
| (1,4,2) | 8 | (8,1,1) | 8 |
| (1,8,1) | 8 | — | — |
| (2,1,1) | 2 | — | — |
| (2,1,2) | 4 | — | — |
| (2,2,1) | 4 | — | — |

- **Cache**：每行组合在 `none` / `tea_cache` / `cache_dit` 中**任选其一**。
- **FP8**：每行在 `off` / `on` 中**任选其一**。
- **CPU Offload**：每行在 `off` / `on` 中任选其一。

以上维度组合后即为“所有可用组合”；下表为**代表性子集**，便于先测主要组合，其余可按同一规则追加行。

### 5.2 测试记录表

**以下为唯一维护的特性测试表**，请直接在本文档中填写与更新。每行一套特性组合。

- **互斥不测**：会互斥的组合不进行测试，表中仅列合法组合。
- **设计**：以 **SP=1 / 2 / 4 / 8** 为基准，分别与每项其它特性组合（基准 + tea_cache / cache_dit / CFG=2 / TP=2 等）；SP×TP×CFG≤8，CPU Offload 仅单卡。
- **首次 Startup(s)**：环境就绪后该配置第一次启动到 Ready 的耗时，单独记录，不参与 mean/std。
- **Stop/Startup/Switch μ、σ**：由第 2～21 次启动的 20 次样本计算（首次不计入）。

| 编号 | GPU | GPU=SP×CFG×TP | Cache     | SP | CFG | TP | CPU Off | FP8 | 首次Startup(s) | Stop μ | Stop σ | Startup μ | Startup σ | Switch μ | Switch σ | 预期 | 备注 |
|-----:|----:|:-------------:|:----------|--:|---:|---:|:--------:|:---:|---------------:|-------:|-------:|----------:|----------:|---------:|---------:|:----:|:-----|
|    1 |   1 | 1=1×1×1       | none      |  1 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 基准 |
|    2 |   1 | 1=1×1×1       | tea_cache |  1 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + Cache |
|    3 |   1 | 1=1×1×1       | cache_dit |  1 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + Cache |
|    4 |   2 | 2=1×2×1       | none      |  1 |   2 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + CFG=2 |
|    5 |   2 | 2=1×1×2       | none      |  1 |   1 |  2 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + TP=2 |
|    6 |   4 | 4=1×1×4       | none      |  1 |   1 |  4 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + TP=4 |
|    7 |   8 | 8=1×1×8       | none      |  1 |   1 |  8 | off      | off |                |        |        |           |           |          |          | PASS | SP=1 + TP=8 |
|    8 |   1 | 1=1×1×1       | none      |  1 |   1 |  1 | on       | off |                |        |        |           |           |          |          | PASS | SP=1 + CPU Offload |
|    9 |   1 | 1=1×1×1       | none      |  1 |   1 |  1 | off      | on  |                |        |        |           |           |          |          | PASS | SP=1 + FP8 |
|   10 |   2 | 2=2×1×1       | none      |  2 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 基准 |
|   11 |   2 | 2=2×1×1       | tea_cache |  2 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 + Cache |
|   12 |   2 | 2=2×1×1       | cache_dit |  2 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 + Cache |
|   13 |   4 | 4=2×2×1       | none      |  2 |   2 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 + CFG=2 |
|   14 |   4 | 4=2×1×2       | none      |  2 |   1 |  2 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 + TP=2 |
|   15 |   8 | 8=2×1×4       | none      |  2 |   1 |  4 | off      | off |                |        |        |           |           |          |          | PASS | SP=2 + TP=4 |
|   16 |   2 | 2=2×1×1       | none      |  2 |   1 |  1 | off      | on  |                |        |        |           |           |          |          | PASS | SP=2 + FP8 |
|   17 |   4 | 4=4×1×1       | none      |  4 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=4 基准 |
|   18 |   4 | 4=4×1×1       | tea_cache |  4 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=4 + Cache |
|   19 |   4 | 4=4×1×1       | cache_dit |  4 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=4 + Cache |
|   20 |   8 | 8=4×2×1       | none      |  4 |   2 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=4 + CFG=2 |
|   21 |   8 | 8=4×1×2       | none      |  4 |   1 |  2 | off      | off |                |        |        |           |           |          |          | PASS | SP=4 + TP=2 |
|   22 |   4 | 4=4×1×1       | none      |  4 |   1 |  1 | off      | on  |                |        |        |           |           |          |          | PASS | SP=4 + FP8 |
|   23 |   8 | 8=8×1×1       | none      |  8 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=8 基准 |
|   24 |   8 | 8=8×1×1       | tea_cache |  8 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=8 + Cache |
|   25 |   8 | 8=8×1×1       | cache_dit |  8 |   1 |  1 | off      | off |                |        |        |           |           |          |          | PASS | SP=8 + Cache |
|   26 |   8 | 8=8×1×1       | none      |  8 |   1 |  1 | off      | on  |                |        |        |           |           |          |          | PASS | SP=8 + FP8 |

（SP=8 时 GPU 已满，不再测 SP=8+CFG=2 或 SP=8+TP>1。）

---

