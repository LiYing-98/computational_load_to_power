# ML.ENERGY Benchmark V3：LLM inference 数据审计（第一阶段）

本仓库只执行任务书中的第一阶段探索：审计 ML.ENERGY Benchmark V3 的**文本 LLM inference** 数据，并判断它是否足以支持后续“任务事前特征 → GPU 能耗/功率”建模。

## 结论先行

公开稳定榜单快照包含 565 条配置级记录、27 个模型、3 个文本任务、2 种 GPU（H100/B200）和 23 个可见字段。它足以支持一个**严格限定在 V3 已观测配置域内**的原型，优先预测 GPU 侧稳态 `energy_per_token_joules` 或聚合 `avg_power_watts`；但它还不足以支持可泛化的“任务事前特征 → 单请求/整任务总能耗”模型。

关键原因是：公开快照没有每次运行的输入长度、计划输出预算、稳态持续时间和稳态总能量；`avg_output_len`、吞吐、ITL、`avg_batch_size` 都是运行后观测，不能作为事前输入。`energy_per_request_joules` 还是由每 token 能耗乘实际平均输出长度估算而来。

## 边界

- 仅含 `gpqa`、`lm-arena-chat`、`sourcegraph-fim` 三个文本 LLM inference 任务。
- 计量口径是稳态窗口内、跨本次运行全部 GPU 求和的 GPU 侧能量与功率；不是单卡值，也不包括 CPU、整机/PDU 或 Node Residual。
- 未下载全量 raw timeline；仓库只保留官方 leaderboard 的三个紧凑 JSON（合计约 0.5 MB）。
- 未涉及 Training、Diffusion、MLLM、CPU、Node Residual，也未训练任何模型。

## 主要交付物

- `reports/mlenergy_v3_llm_audit.html`：主报告（自包含 HTML）。
- `docs/field_dictionary.md`：完整字段字典、事前输入/目标/泄漏/诊断分级。
- `notebooks/mlenergy_v3_llm_audit.ipynb`：已执行的可复现审计笔记本。
- `scripts/audit_mlenergy_v3_llm.py`：仅用 Python 标准库的审计脚本。
- `analysis/audit_results.json`：机器可读审计结果。
- `data/compact/llm/*.json`：官方 leaderboard 稳定记录的紧凑快照。

## 复现

在仓库根目录运行：

```powershell
python scripts/audit_mlenergy_v3_llm.py
python -m unittest discover -s tests -v
```

## 数据来源与版本

紧凑快照来自 [ML.ENERGY leaderboard](https://github.com/ml-energy/leaderboard) 提交 `5d899ea5a1c8934c0eff691df2b731d20490bb2a`，其 `index.json` 标记更新时间为 2026-02-16。字段口径以 [ml-energy/data](https://github.com/ml-energy/data) 提交 `aaf4d45b8289d96c56c0fb1ebf86dd4d57fac66c` 和 [ml-energy/benchmark](https://github.com/ml-energy/benchmark) 提交 `c1d6557fbf7c7c495749b0fcf649f71774eabde9` 为准。

Hugging Face 上的 `runs/llm.parquet` 当前需要登录并接受数据集访问条件；本次环境没有授权令牌，因此没有绕过门控。报告对“公开紧凑快照审计”和“gated parquet 才能补做的检查”作了明确区分。
