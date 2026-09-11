# ML.ENERGY Phase 1.5：完整汇总与工作量字段审计

本目录承接 Phase 1，只补齐两类证据：

1. gated `runs/llm.parquet` 的完整字段、稳定/不稳定规模、重复实验和能量—功率—时间口径；
2. 约 6–12 个代表性 LLM inference run 在内嵌 timeline 之前的结果/配置头部，用于判断真正可在执行前获得或推导的工作量字段。

## 严格边界

- 仅研究 ML.ENERGY Benchmark V3 文本 LLM inference。
- 不下载全量 raw timeline 或 Prometheus/GPU power timeline。
- 不训练任何预测模型。
- 不涉及 Training、Diffusion、MLLM、CPU、Node Residual、NLR 或 CMU/LBNL。
- 不下载模型权重，不猜测板卡规格或节点拓扑。
- Hugging Face token 不写入任何文件或 Git。

## 数据保管

`step1/data/runs/` 与 `step1/data/selected_runs_raw/` 是本机审计缓存，已由仓库根目录 `.gitignore` 排除。代表性 run 仅保留 timeline 键之前的 16.6 KB 元数据前缀。Git 中只保存字段级统计、哈希、来源、选择清单和可复现代码，不保存 gated 原始记录。

## 预期交付

- `analysis/`：机器可读审计结果与 selected-run 清单；
- `docs/`：完整字段字典、任务工作量矩阵和 `X_pre` 目录；
- `notebooks/`：已执行的复现笔记本；
- `report_app/dist/index.html`：本地技术报告；
- `reports/`：简明结论与验证收据。

最终只给出 A/B/C 阶段判定，然后停止。
