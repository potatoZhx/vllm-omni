# 离散事件模拟器设计说明

## 快速开始

```bash
cd simulation
python simulation.py
```

**配置**：`config/simulation_config.yaml`（主配置，含 rps、worker、算法、profile 路径、请求 mix 等）。

**输出**：`output/{profile名}/` 下生成各算法的 JSON 与 CSV，跑完自动画图到同目录。

---

## 一、设计思路（放前面）

### 目标

在**不实际运行算法、不发起真实压力测试**的前提下估计算法性能，构建一个**离散事件模拟器**。

模拟器不执行真实推理，只依赖一张预先测得的执行时间表（如 `profile_*.csv`），在此基础上模拟调度器与多个 worker 的行为，得到吞吐、延迟、P50/P95/P99、完成请求数、失败请求数、SLO 达成率等指标。指标口径与 `benchmarks/diffusion` 及 `diffusion_bench/plot_results.py` 对齐，便于直接用于画图与对比。

### 核心建模假设

1. **忽略通信与调度开销**：请求到达调度器、调度器将请求放入 worker 队列、worker 返回结果均不耗时；请求一旦被调度，即在同一时刻进入目标 worker 队列。
2. **服务时间由 profile 查表得到**：每个请求在某个 worker 上的执行时间由 profile 表查表得到，不做真实推理；查不到则报错。
3. **固定模拟时长作为注入结束条件**：设总模拟时长为 `T_end`。调度器仅在下一次发送时点 `<= T_end` 时继续发送请求；超过 `T_end` 后停止注入，并将调度器时点置为 `inf`。之后系统继续运行直至所有 worker 空闲且队列为空，模拟结束。

### 实现原则（与项目契合）

- **配置与代码分离**：全局参数、worker 列表、算法列表、输入输出路径等均在 YAML 中配置，模拟器只读配置。
- **查表严格**：size 或 config 在 profile 中无匹配即报错；steps 按约定档位映射（4–6→5，25–35→30，40–50→50）。
- **调度可插拔**：仅替换调度模块即可切换算法，其他模块（事件循环、worker、队列、统计、输出）保持不变。
- **输出即画图输入**：输出 JSON 带 `algorithm` 且字段名与 `diffusion_bench/plot_results.py` 一致，支持单文件多算法或每算法一文件、跑完后可自动画图或手动画图；自动画图时**一指标一图**，每图内为各算法的曲线，避免多指标挤在一张图里。

### 设计小结

- **配置**：YAML 统一管理；worker 用 (sp, cfg, tp) 三整数，程序拼接为 config_id；算法列表可配置并带注释命名。
- **查表**：size、config_id 必须与 profile 一致，steps 按档位映射；查不到即报错；不输出显存等无关字段。
- **算法**：与 global_scheduler 一致（round_robin、fcfs、short_queue_runtime、estimated_completion_time），可插拔；首推实现 round_robin。
- **输出**：必含 `algorithm`，与 plot_results 对齐；多算法可多 JSON 或单 JSON + `group-by algorithm`；SLO 默认 slo_scale=3；跑完后可配置自动画图（默认 P95 与平均延迟，一指标一图）。

---

## 二、工程化表述（具体约定与用法）

### 1. 配置与全局参数（仅 YAML）

**所有全局配置均放在 YAML 配置文件中，不写在模拟器代码里。**

配置文件应至少包含：

- **模拟参数**：模拟时长 `T_end`、**rps 列表**（如 `[0.5, 1.0, 1.5]`；对每个算法在每个 rps 下各跑一轮，得到横轴为 rps 的曲线数据）、SLO 相关（见下）。
- **Worker 列表**：每个 worker 用三个整数 `sp`、`cfg`、`tp` 表示，程序内部拼接为完整 config_id（见下节）。
- **调度算法**：要参与测试的算法列表（可多选）；调度逻辑为可插拔模块。
- **输入/输出**：profile 路径、**请求数据集来源与 trace/random 配置**、输出目录；多算法时可指定输出多个 JSON 或单 JSON 带 `algorithm` 字段。
- **画图**（可选）：跑完后是否自动画图、默认画图指标（每个指标单独一张图）、图保存目录与前缀。

**SLO**：`slo_scale` 默认 **3**（即 `slo_ms = 预估执行时间_ms * slo_scale`）。与画图无关的字段（如显存占用）不输出。

---

### 请求数据集与 random-request-config 等价物

#### 数据集来源选择（dataset）

在 `simulation.simulation` 段中，可通过 `dataset` 字段选择请求数据集来源：

- `trace`（默认行为）：  
  - 若 `trace_path` 存在，则从 trace 文件解析出一系列 `(size, steps)`，**仅用于请求类型顺序**；  
  - 注入时刻仍为 `t = 0, 1/rps, 2/rps, ...`，trace 中的 timestamp 不使用。
- `default`：  
  - 忽略 `trace_path`，所有请求都使用 `default_request` 中配置的单一 `(size, steps, num_frames, task_type)`。
- `random`：  
  - 使用 `random_request_config` 中配置的多种请求类型，并按 `weight` 做加权随机采样，模拟 `benchmarks/diffusion/diffusion_benchmark_serving.py` 中的 `--random-request-config` 行为。

若未显式配置 `dataset`，则保持兼容旧行为：若 `trace_path` 存在等价于 `trace`，否则等价于 `default`。

#### random_request_config：对齐 benchmark 的 random 数据集

当 `dataset: random` 时，simulation 使用 `simulation.random_request_config` 作为“请求类型池”：

```yaml
simulation:
  dataset: "random"
  random_request_config:
    - { width: 512,  height: 512,  steps: 20, num_frames: 1, task_type: "image", weight: 0.15 }
    - { width: 768,  height: 768,  steps: 20, num_frames: 1, task_type: "image", weight: 0.25 }
    - { width: 1024, height: 1024, steps: 25, num_frames: 1, task_type: "image", weight: 0.45 }
    - { width: 1536, height: 1536, steps: 35, num_frames: 1, task_type: "image", weight: 0.15 }
  random_request_seed: 42
```

- **与 benchmark 对齐**：使用 `rng.choices(config, weights, k=N)` 一次性预采样 N 个 profile（N = ceil(t_end * rps)），与 `diffusion_benchmark_serving.py` 的 `RandomDataset` 一致；  
- 到达时间 `t = 0, 1/rps, 2/rps, ..., (N-1)/rps`，与 `iter_requests` 一致；  
- `t_end` 应对齐 benchmark 的 `NUM_PROMPTS_DURATION_SECONDS`（如 1800）；  
- `steps` 与 `num_inference_steps` 等效，兼容 benchmark 的 `--random-request-config`；  
- 请求的执行时间通过 profile 查表得到。

这样可以在仿真层面构造与 benchmark **完全一致**的请求序列与到达模式。

详见 [BENCHMARK_ALIGNMENT.md](BENCHMARK_ALIGNMENT.md) 地毯式核对报告。

**统计口径**（与 `diffusion_benchmark_serving.calculate_metrics` 对齐）：

- `duration`：首请求到达至末请求完成的时长
- `throughput_qps`：completed_requests / duration
- `latency_*` / `waiting_time_*` / `service_time_*` / `slo_attainment_rate`：基于**全部已完成请求**（不按窗口过滤），再配合 `profile_source=json + profile_path=../profile/qwen_profile_random.json` 复用真实测得的运行时间。

---

### 2. Worker 配置与 profile 查表

#### Worker 配置表示

每个 worker 的配置在 YAML 中由 **三个整数** 表示：

- `sp`（如 1, 2, 4, 8）
- `cfg`（如 1, 2）
- `tp`（如 1, 2, 4, 8）

程序内部将三者拼接为 **完整 config 标识**，与 profile 表头一致，例如：

- `sp=1, cfg=1, tp=1` → `sp1_cfg1_tp1`
- `sp=2, cfg=1, tp=2` → `sp2_cfg1_tp2`

即格式为：`sp{sp}_cfg{cfg}_tp{tp}`。profile 表中的 `config_id` 列即为此标识。

#### Profile 表结构（示例：profile_qwen.csv）

| 列名             | 含义                                                         |
|------------------|--------------------------------------------------------------|
| `size`           | 请求分辨率，如 `128x128`、`1024x1024`                        |
| `steps`          | 推理步数（表中为离散档位：1, 5, 10, 30, 50）                 |
| `config_id`      | 与 worker 的 sp/cfg/tp 拼接结果一致，如 `sp1_cfg1_tp1`       |
| `request_time_s` | 该 (size, steps, config_id) 下的执行时间（秒）               |

#### 查表规则（严格匹配，查不到即报错）

查询每个请求的 `request_time_s` 时：

1. **config_id**：必须与目标 worker 的配置一致（由该 worker 的 sp/cfg/tp 拼接得到）。
2. **size**：必须与请求的 size 完全一致；若表中不存在该 size，**直接报错**，不做插值或默认值。
3. **steps**：请求的 `num_inference_steps` 按以下规则映射到表中的离散档位后再查表：
   - **1** → 按 **1** 查表
   - **2 ≤ steps ≤ 6** → 按 **5** 查表
   - **7 ≤ steps ≤ 24** → 按 **10** 查表
   - **25 ≤ steps ≤ 35** → 按 **30** 查表
   - **36 ≤ steps ≤ 50** → 按 **50** 查表  
   其他步数若表中无对应档位，则按实现约定或报错。

若 **size 或 config_id 在 profile 中无匹配行**，必须 **报错退出**，不允许静默回退或猜值。

---

### 3. 调度算法：可插拔与对齐 global_scheduler

模拟器要对 `tests/global_scheduler`（及 `vllm_omni.global_scheduler.policies`）中的 **每个算法** 提供支持，且为 **可插拔**：

- 仅 **更换调度器所用的调度模块**（即“选哪个算法”），事件循环、worker、队列、统计等 **其余模块不改**。
- 通过配置（YAML）选择当前运行的算法；可配置 **要测试的算法列表**，对每个算法在 rps 列表的每个取值下各跑一轮模拟，输出供画图（横轴 rps，每算法一条曲线）。

支持的算法名称与 global_scheduler 保持一致，便于一一对应。**无论采用哪种算法，每个到达的请求都会被立即派发到某一实例**：调度器只做“选哪台”的决策，不会因为实例都忙而拒绝或延迟派发；请求会进入目标实例的队列等待执行。因此 **arrival_time 表示请求到达调度器并被派发（入队）的时刻**，不会失真；请求可能在该时刻之后才开始执行（start_time），二者之差即为排队等待时间（waiting_time）。

各算法含义如下（均为“选哪个实例”，选完后请求进入该实例队列）：

| 配置名 | 含义（简要） |
|--------|----------------|
| `round_robin` | 轮询 |
| `fcfs` | 先到先服务 |
| `short_queue_runtime` | 最短队列预估时间 |
| `estimated_completion_time` | 预估完成时间最小 |

**算法说明（详细）：**

- **round_robin（轮询）**  
  维护一个游标，按实例列表顺序依次选择下一个实例。  
  - 若有空闲实例：只在“当前空闲”的实例集合内轮询，选下一个空闲的。  
  - 若**全部忙碌**：仍会选一个实例（按游标选），请求进入该实例的队列排队；游标照常前移，下次请求换到下一个实例，从而在负载高时也尽量均匀分摊到各实例，不把请求堆在某一台上。  
  因此“该发请求时没有实例空闲”时依然会发，只是请求会排队；arrival_time 仍是本次派发时刻。

- **fcfs（先到先服务）**  
  优先选“当前空闲”的实例（inflight < 该实例并发上限）；若有多个空闲，取配置中**排在前面**的第一个空闲实例。  
  - 若**全部忙碌**：退化为选当前负载最小的实例（inflight 最小）；相同时按 tie_breaker（random 或 lexical）打破平局。请求进入该实例队列。  
  同样，请求一定会被派发，arrival_time 为派发时刻。

- **short_queue_runtime（最短队列预估时间）**  
  选**当前剩余总工作量（秒）**最小的实例。剩余工作量 = 正在执行任务的剩余时间 + 队列中所有等待请求的 service_time 之和（对队列内每个请求按 size/steps 与实例 config 查表）。与“队列长度×本请求时间”的近似不同，本实现基于**剩余工作量**，适合异构与混合请求。

- **estimated_completion_time（预估完成时间最小，ECT）**  
  选**本请求在该实例上的预估完成时间**最小的实例。预估完成时间 = 该实例当前剩余总工作量（同上）+ 本请求在该实例的 service_time。即让本请求尽量在“能最早完成”的实例上排队。

在 YAML 的算法参数配置部分，应为每个可选项添加 **注释标出算法名称**，避免与实现或 global_scheduler 的命名对不上号。

**入队与队列**：请求被派发到某实例后放入该实例队列**末尾**；**入队后、worker 立即派工检查之前**保留“调度器对队列排序”的扩展点，当前为占位，以后可在此扩展。默认行为仅为将任务放到队列末尾。

**配置示例（片段）：**

```yaml
scheduler:
  algorithm: round_robin
  algorithms_to_run: [round_robin, fcfs]
```

---

### 4. 输出格式与画图对接

#### 时间口径（request_time_s 与请求三时点）

- **profile 表中的 `request_time_s`**：仅表示该请求在 worker 上**从开始执行到执行结束**的时间（即服务时间，不含排队）。
- **请求发起时间**：请求到达调度器的时刻（`arrival_time`）；此时请求不一定立即被执行，可能进入某 worker 队列等待。
- **开始执行时间**：worker 从队列取出该请求并开始执行的时刻（`start_time`）。
- **完成时间**：该请求执行完毕的时刻（`finish_time`）。  
因此：`latency = finish_time - arrival_time`，`waiting_time = start_time - arrival_time`，`service_time = finish_time - start_time`（与 profile 查表得到的 `request_time_s` 一致）。

#### 单次运行输出（单算法）

每次运行输出的 JSON 必须包含 **算法字段**，以便与 `diffusion_bench/plot_results.py` 对接：

- 必含字段：**`algorithm`**（字符串，与配置中的算法名一致，如 `round_robin`、`fcfs`）。
- 与画图脚本对齐的汇总字段（命名保持一致）：  
  `rps`、`duration`、`completed_requests`、`failed_requests`、`throughput_qps`、  
  `latency_mean`、`latency_median`、`latency_p50`、`latency_p95`、`latency_p99`、  
  `waiting_time_mean`、`waiting_time_p95`、`waiting_time_p99`、  
  `service_time_mean`、`service_time_p95`、`service_time_p99`、  
  `slo_attainment_rate`（若启用 SLO）等。
- **统计与请求明细分两个文件**：  
  - **统计文件**（如 `round_robin.json`）：每个 rps 一条汇总记录，含 `algorithm`、`rps` 及各项汇总指标，不含每请求明细，供画图与对比。  
  - **请求明细文件**（如 `round_robin_requests.csv`）：**CSV 格式**，每行一个已完成请求，列包括 `algorithm`、`rps`、`request_id`、`assigned_worker_index`、`assigned_worker_config`、`arrival_time`、`start_time`、`finish_time`、`latency`、`waiting_time`、`service_time`、`size`、`steps`，便于用表格工具查看与筛选。

全部输出指标字段见 `simulation_config.yaml` 内注释。输出中不包含与画图无关的字段（如显存占用等），直接舍弃。

#### 多算法时的输出与画图对接

- **方式 A**：每个算法两个文件——统计 `round_robin.json`、`fcfs.json` 等（数组，每 rps 一条汇总）；请求明细 `round_robin_requests.csv`、`fcfs_requests.csv` 等（CSV，每行一个请求）。画图使用统计文件：`plot_results.py --input-dir <输出目录> --algorithms round_robin fcfs ...`。
- **方式 B**：配置 `output_merged: true` 时额外生成 `merged.json`（统计）、`merged_requests.csv`（请求明细，CSV）；画图使用 `plot_results.py -i output/merged.json --split-by-type --group-by algorithm`。

**与 plot_results 对接示例：**

- 多算法各输出一个 JSON 到目录 `output/`，画图时：
  ```bash
  python diffusion_bench/plot_results.py --input-dir output --algorithms round_robin fcfs -o figs/compare --split-by-type
  ```
- 或合并为单 JSON（每条记录含 `algorithm`），画图时：
  ```bash
  python diffusion_bench/plot_results.py -i output/merged.json --split-by-type --group-by algorithm -o figs/compare
  ```

**自动画图**：跑完后若配置中 `plot.after_run: true`，会自动调用 `diffusion_bench/plot_results.py` 画图并保存。默认画图指标为 `latency_p95`、`latency_mean`，可在 YAML 的 `plot.metrics` 中修改；**每个指标单独一张图**，每图内为各算法的曲线。因画图脚本使用 `split_by_type`，实际生成的文件名会带类型后缀，例如 `compare_latency_p95_latency.png`、`compare_latency_mean_latency.png`（而非 `compare_latency_p95.png`），与 `plot_results.py` 的命名规则一致。

---

### 5. 系统组成与数据结构

- **Actor**：调度器（scheduler）、若干 worker；系统维护全局时钟 `global_time`。
- **调度器状态**：`scheduler_next_time`（下一次发送请求的时点）、`rps`、当前选中的 **调度模块**（可插拔）。
- **Worker 状态**：每个 worker 具有 (sp, cfg, tp) 及拼接得到的 config_id、一个任务队列、`worker_next_time`（下一次完成事件时点，空闲为 `inf`）。
- **请求状态**：请求 ID、请求类型（含 size、steps）、到达时间、开始执行时间、完成时间、分配到的 worker、目标 SLO（若启用）。

---

### 6. 事件与单轮流程

每轮流程固定如下：

1. **选择下一个 actor**：在调度器下一次发送时点与各 worker 下一次完成时点中取最小者；对应时点最小者即为当前执行 actor。
2. **执行事件并推进时钟**：若为调度器则发送一个请求并按当前调度模块选 worker 入队；若为 worker 则完成当前请求并更新统计。两者都先用自身时点更新 `global_time`。
3. **入队后、立即派工前**：调度器对队列的排序为占位（当前默认仅将任务放到队列末尾，以后可在此扩展）。此步不推进全局时钟。
4. **立即派工检查**：在全局时钟更新后、进入下一轮选择前，对每个 worker 检查：若当前空闲且队列非空，则从**队首**取请求、查表得到 `request_time_s`、安排完成事件。此步不推进全局时钟。
5. **结束条件**：调度器时点为 `inf`、所有 worker 时点为 `inf`、所有队列为空时，模拟结束。

---

### 7. 指标统计与口径

- **基础时间字段**：对每个请求记录 `arrival_time`、`start_time`、`finish_time`；由此得到 `latency`、`waiting_time`、`service_time`。
- **汇总指标**：与 benchmark/画图对齐，包括 `duration`、`completed_requests`、`failed_requests`、`throughput_qps`、`latency_mean`、`latency_median`、`latency_p50`、`latency_p95`、`latency_p99`、`waiting_time_*`、`service_time_*`、`slo_attainment_rate` 等；不含显存等与画图无关字段。完整列表见配置文件内注释。

---

### 8. 使用方式（单 YAML + 单 Python）

**运行前检查**：已安装 `pyyaml`；配置文件中的 `profile_path` 指向的 profile 文件存在（CSV 需含列 `size, steps, config_id, request_time_s`；JSON 需含 `profiles` 列表）；workers 的 (sp, cfg, tp) 或 `instance_type` 在 profile 中有对应项；默认请求的 size/steps 在 profile 中有对应行。若使用 trace（如 `sd3_trace_redistributed.txt`），将 `trace_path` 设为该文件路径（相对本 yaml 所在目录）。trace 格式为每行 `Request(..., height=..., width=..., num_inference_steps=...)`；**模拟器只用到请求顺序对应的 size、steps**；**请求发送唯一逻辑是 rps**：调度器发起第 i 个请求时自己的时点即为该请求的 `arrival_time`（= i/rps），**数据集中的 timestamp 不使用**。其余字段（prompt、negative_prompt、timestamp 等）不解析、不使用。

**依赖**：`pip install pyyaml`（自动画图需在项目根目录运行并已安装 matplotlib）。

**目录结构**：

```
simulation/
├── config/          # YAML 配置文件（profile_path/output_dir/trace_path 均相对 config/ 解析）
├── profile/         # profile 数据（JSON/CSV）
├── tools/           # profile 转换工具（csv_to_profiles.py 等）及源数据
├── output/          # 仿真输出
└── simulation.py
```

**配置文件**：`config/simulation_config.yaml` 等（其中 `profile_path`、`output_dir`、`trace_path`、`plot.output_dir` 相对该 yaml 所在目录 `config/` 解析）。

**运行**（在项目根目录 `vllm-omni` 下）：

```bash
python simulation/simulation.py simulation/config/simulation_config.yaml
```

或在 `simulation/` 目录下：

```bash
cd simulation && python simulation.py config/simulation_config.yaml
# 或省略 config 参数，默认使用 config/simulation_config.yaml
python simulation.py
```

**输出**：

- 默认在配置的 `output_dir` 下每个算法生成两个文件：**统计** `{algo}.json`（每 rps 一条汇总）、**请求明细** `{algo}_requests.csv`（CSV，每行一个请求）。
- 若配置中 `output_merged: true` 则额外生成 `merged.json`（统计）、`merged_requests.csv`（请求明细，CSV）。
- 若 `plot.after_run: true`，则在同一目录（或 `plot.output_dir`）下生成按指标命名的图（一指标一图，前缀由 `plot.output_prefix` 指定）。实际文件名会带类型后缀，例如 `compare_latency_p95_latency.png`、`compare_latency_mean_latency.png`。

**手动画图**（多算法对比）：

```bash
python diffusion_bench/plot_results.py --input-dir simulation/output --algorithms round_robin fcfs short_queue_runtime -o figs/compare --split-by-type
```

---

### 9. 第一版实现约束

以下约束明确第一版的边界，避免首版过度复杂；后续版本可再放宽。

- **请求注入**：请求发送按**时间驱动**：到达时刻 t = 0, 1/rps, 2/rps, …，**当 t ≤ T_end 时**生成该请求（与“下一次发送时点 ≤ T_end 继续发送”一致）。例如 rps=0.1、T_end=200 时为 21 个请求（t=0,10,…,200）。第 i 个请求的 `arrival_time = i/rps`。trace 仅提供请求顺序对应的 size、steps，**数据集中的 timestamp 不使用**。
- **查表**：第一版仅支持 **profile 严格查表**（size、config_id、steps 档位均需命中），**不支持**插值、外推或默认回退。
- **失败机制**：第一版**不模拟**超时、队列溢出等失败；统计中 **failed_requests 固定为 0**，接口预留即可。
- **输出方式**：统计与请求明细分两个文件（统计 JSON + 请求明细 CSV，如 `round_robin_requests.csv`）；可选合并为 `merged.json` / `merged_requests.csv` 由配置控制。
- **调度算法**：第一版**首个实现**的算法为 **round_robin**，其余算法在此基础上按同一接口可插拔扩展。
