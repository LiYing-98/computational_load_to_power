# 数据来源、审计范围与证据边界

## 本次实际使用的数据

| 文件 | 记录数 | SHA-256 |
|---|---:|---|
| `data/compact/llm/gpqa.json` | 187 | `640e92a4bdf88512840f8369febca5bf26218fade225e7184458fa1035e41371` |
| `data/compact/llm/lm-arena-chat.json` | 328 | `1461ab6d008e4ee06c1c7ed03a308c437b0dadbee4b4042103120723bb059426` |
| `data/compact/llm/sourcegraph-fim.json` | 50 | `c4de9f316e6d30984fdc02a627f33007484bac390bbc37fca5b3f8a3d3ad116b` |

来源是官方 leaderboard 提交 `5d899ea5a1c8934c0eff691df2b731d20490bb2a`。其 `data/index.json` 标记 `last_updated=2026-02-16`。三个文件是官方 `LLMRuns.from_hf()` 默认稳定筛选后再裁剪字段形成的展示快照。

## 明确没有获取的数据

- 未下载任何 raw `results.json`、Prometheus 时间线或 GPU telemetry timeline。
- 未取得 Hugging Face gated `runs/llm.parquet`，因为执行环境没有已授权令牌，也没有替用户接受访问条件。
- 未使用 MLLM 的 `image-chat`/`video-chat`、Diffusion、Training、CPU、整机/PDU 或 Node Residual 数据。

因此，本次对 23 个公开字段完成了实际值审计；对 parquet 另外 14 个字段完成的是**官方定义与可用性审计**，不是分布或缺失率审计。

## 计量口径

官方 benchmark 先按请求流进入稳定状态后开始窗口，在大部分请求结束、进入回落前停止窗口；随后把该窗口内 `gpu_energy` 字典中全部 GPU 的能量相加。因而：

1. `steady_state_energy_joules` 是本次运行全部 GPU 的聚合 GPU 侧能量。
2. `avg_power_watts = steady_state_energy_joules / steady_state_duration_seconds`，也是全部 GPU 聚合功率。
3. 这不是单卡功率；跨 `num_gpus` 直接比较会混入口径尺度差异。
4. 这不是 CPU、节点输入电力或 PDU 口径，不能用它推断 Node Residual。
5. 官方实现中没有看到空闲基线扣除，因此应理解为稳态窗口内 GPU 总能量，而非净增量能量。

## 稳定筛选

官方 data 工具会把稳态时长小于 20 秒、每 token 能耗无效/非正、或 `avg_batch_size / max_num_seqs < 0.85` 的运行标为不稳定；同一模型/任务/GPU/卡数下，一旦较小 batch 不稳定，更大 batch 会级联标为不稳定。leaderboard 默认只发布稳定运行，但裁剪后的公开 JSON 不保留 `is_stable` 与原因。

本次公开快照中 565/565 条 `avg_batch_size / max_num_seqs >= 0.85`，最低为 0.8540；这只能复核公开可见的这一项，不能重新执行完整稳定性筛选。

## 图表映射

| 图表 | 分析问题 | 形式 | 编码 | 来源 |
|---|---|---|---|---|
| 任务 × GPU 覆盖 | 三个任务在 H100/B200 上是否均衡 | 分组柱状图 | x=task, y=记录数, color=GPU | 三个 compact JSON |
| GPU 卡数覆盖 | 1/2/4/8 卡配置覆盖如何 | 柱状图 | x=num_gpus, y=记录数 | 三个 compact JSON |

图表只表达覆盖，不把配置条数误写成独立请求样本数。

## 主要官方来源

- [ML.ENERGY Benchmark V3 数据集页](https://huggingface.co/datasets/ml-energy/benchmark-v3)
- [ML.ENERGY records API](https://ml.energy/data/api/records/)
- [ML.ENERGY data guide](https://ml.energy/data/guide/)
- [官方 data 工具](https://github.com/ml-energy/data/tree/aaf4d45b8289d96c56c0fb1ebf86dd4d57fac66c)
- [官方 benchmark](https://github.com/ml-energy/benchmark/tree/c1d6557fbf7c7c495749b0fcf649f71774eabde9)
- [官方 leaderboard](https://github.com/ml-energy/leaderboard/tree/5d899ea5a1c8934c0eff691df2b731d20490bb2a)
