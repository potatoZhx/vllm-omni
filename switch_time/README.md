# switch_time：配置切换与性能评测脚本

在 **vLLM-Omni** 上测试「不同并行配置（SP/CFG/TP 等）下服务器启停与推理耗时」，不修改仓库源码，仅通过 `vllm serve` CLI 与 HTTP 接口完成。适用于 Qwen-Image 与 Wan2.2 等扩散模型。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| **run_switch_all.sh** | 特性切换耗时：26 种配置（含 cache/FP8/cpu_offload 等）**分开测**，每配置「1 次首次启动 + 10 次停→起」。不发起推理，仅轮询 `/v1/models` 判就绪，输出 Stop/Startup/Switch 均值与标准差。 |
| **run_switch_parallel.sh** | Qwen-Image **并行表**：16 种 GPU=SP×CFG×TP 组合，每配置「1 次首次 + 5 次停→起」。不推理，仅轮询就绪。输出 CSV 与统计到 `qwen_parallel_log/`。可 `sbatch` 或直接 `bash`。 |
| **run_switch_parallel_wan.sh** | Wan2.2 **并行表**：23 种配置（TP 套件 + Ring 冒烟 + HSDP P0），逻辑同 `run_switch_parallel.sh`，模型为 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`，日志到 `wan_parallel_log/`。 |
| **test_qwen_profile.sh** | Qwen-Image **文生图性能画像**：按 `profile.md` 的 20 个数据项（分辨率×步数）× 9 种配置，对每个组合发 5 次真实 `/v1/images/generations` 请求，记录单次完成时间与 run2~5 均值。结果追加到 `logs/` 下 CSV 与结果日志。 |
| **profile.md** | Qwen-Image 实验说明、数据项/配置定义、**实验结果记录**（请求完成时间均值、冷启动、run1 首请求等表格）。 |
| **Qwen-Image-特性切换测试说明表.md** | Qwen-Image 特性切换测试的说明与结果表（与 profile 互补）。 |
| **并行测试表_qwen.md** | 汇总 run_switch_parallel.sh 的 **16 配置** 结果：首次 Startup、Stop/Startup/Switch 的 μ 与 σ。 |
| **并行测试表_wan.md** | 汇总 run_switch_parallel_wan.sh 的 **多配置** 结果：同上。 |
| **image/** | 结果图表（如特性切换结果表相关图片）。 |
| **logs/** | 各脚本生成的运行日志、CSV、server 日志（可按需加入 .gitignore）。 |

脚本均依赖同目录或上级的 **setup_env.sh**（若存在）做环境准备；支持 `START_CONFIG`、`START_DATA_ITEM` 等环境变量续跑或跳过部分配置/数据项。

---

## 使用方式

- 在 `switch_time` 目录下激活 conda 后执行：`bash run_switch_parallel.sh` 或 `sbatch run_switch_parallel.sh`（其它脚本同理）。
- 可选环境变量见各脚本头部注释（如 `START_CONFIG=8` 从第 8 个配置起测）。
