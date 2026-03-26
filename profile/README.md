# Runtime Profile Format

这个目录用于存放实例内调度器读取的 runtime profile JSON 文件。

当前用途：
- `sjf` 调度中的任务时长估算
- `slo_first` 调度中的任务时长估算

调度器会优先读取这里的 profile 数据；如果没有命中，再退回到代码里的启发式估算。

## 使用方式

启动服务时传入：

```bash
vllm serve Qwen/Qwen-Image --omni \
  --instance-scheduler-policy sjf \
  --instance-runtime-profile-path ./profile \
  --instance-runtime-profile-name qwen-image-sp2-cfg2
```

参数说明：
- `--instance-runtime-profile-path`
  - 可以是单个 JSON 文件
  - 也可以是一个目录；目录下所有 `*.json` 都会被扫描
- `--instance-runtime-profile-name`
  - 可选
  - 用于按 `instance_type` 过滤 profile 记录
  - 当前是字符串精确匹配，不做模糊匹配或字段拆解

## JSON 格式

支持三种顶层格式：

1. 顶层对象，字段名为 `profiles`
2. 顶层对象，字段名为 `entries`
3. 顶层直接是数组

推荐格式：

```json
{
  "profiles": [
    {
      "instance_type": "qwen-image-sp2-cfg2",
      "task_type": "image",
      "width": 1024,
      "height": 1024,
      "num_frames": 1,
      "steps": 10,
      "latency_ms": 420
    }
  ]
}
```

## 字段说明

每条记录支持这些字段：

- `instance_type`
  - 可选
  - 用于标识实例配置，例如：
    - `qwen-image-tp1`
    - `qwen-image-sp2-cfg2`
    - `wan-video-sp4-cfg2`
  - 如果启动时传了 `--instance-runtime-profile-name`，这里必须精确一致才会命中

- `task_type`
  - 必填
  - 取值：
    - `image`
    - `video`

- `width`
  - 必填
  - 请求宽度

- `height`
  - 必填
  - 请求高度

- `resolution`
  - 可选
  - 当 `width/height` 没写时，可用 `resolution` 代替，等价于正方形分辨率

- `num_frames`
  - 可选
  - 图片请求默认按 `1` 处理
  - 视频请求建议显式填写

- `steps`
  - 必填
  - 对应 profiling 时测得的 step 数
  - 当前调度器会优先使用这些离散点做插值或比例外推

- `latency_ms` / `latency_s`
  - 二选一
  - 表示该配置下该任务的运行时间

## 估算规则

### 1. 精确命中

如果以下维度全部命中：
- `task_type`
- `width`
- `height`
- `num_frames`
- `steps`

则直接使用 profile 中的时长。

### 2. steps 插值 / 外推

如果 `task_type + width + height + num_frames` 命中，但 `steps` 不是 profile 中的离散点：
- 在相邻 step 点之间做线性插值
- 小于最小 step 或大于最大 step 时，按最近点做比例外推

因此你现在的 profiling 只测 `10 / 30 / 50 steps` 是可以直接支持的。

### 3. 最近 profile 缩放

如果 `steps` 以外的维度不完全命中：
- 会在同一 `task_type` 下选最近的 profile 组
- 再按分辨率面积和帧数做缩放

这一步是 fallback，不如精确 profile 稳定。

### 4. 启发式 fallback

如果没有可用 profile：
- 调度器会退回到当前代码中的启发式估算
- 不会阻塞请求执行

## 图片与视频的建议写法

### 图片

建议至少覆盖：
- 常用分辨率
- `10 / 30 / 50 steps`

示例：

```json
{
  "instance_type": "qwen-image-sp2-cfg2",
  "task_type": "image",
  "width": 1024,
  "height": 1024,
  "num_frames": 1,
  "steps": 30,
  "latency_ms": 1180
}
```

### 视频

建议至少覆盖：
- 常用分辨率
- 常用帧数
- `10 / 30 / 50 steps`

示例：

```json
{
  "instance_type": "wan-video-sp2-cfg2",
  "task_type": "video",
  "width": 1280,
  "height": 720,
  "num_frames": 16,
  "steps": 30,
  "latency_ms": 4960
}
```

## 推荐命名

建议把实例配置完整写进 `instance_type`，例如：

- `qwen-image-tp1`
- `qwen-image-sp2-cfg2`
- `wan-video-sp4-cfg2`
- `wan-video-sp4-cfg2-hsdp2`

当前实现不会解析 `tp/sp/cfg`，只做精确字符串匹配。

## 参考文件

仓库里提供了一份样例：

- [instance_runtime_profile.sample.json](./instance_runtime_profile.sample.json)

后续如果 profiling 脚本正式产出 JSON，建议直接对齐这份 README 的 schema。
