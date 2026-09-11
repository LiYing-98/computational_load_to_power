# ML.ENERGY V3 LLM 字段字典与建模分级

## 分级规则

本表按“预测发起之前能否获知”分级，而不是按“数据里是否有这一列”分级。

- **候选事前输入**：任务提交与部署配置确定后、运行前即可获知。
- **目标**：要预测的 GPU 侧结果；永远不能同时作为输入。
- **泄漏/运行后结果**：只有执行后才能观测，不能进入事前特征。
- **分组/诊断**：用于去重、质量控制、划分或追溯，不建议作为主模型输入。
- **标识泄漏风险**：虽然运行前已知，但可能让模型记住具体模型，掩盖跨模型泛化能力。

`公开`表示本次使用的稳定 leaderboard 紧凑 JSON 是否暴露该字段。完整 `runs/llm.parquet` 的字段定义来自官方 `LLMRun` 数据结构；本次未取得 gated parquet，故对其未公开字段只做口径审计，不声称完成分布审计。

## 字段表

| 字段 | 公开 | 官方含义/计算口径 | 本阶段分级 | 使用意见 |
|---|---:|---|---|---|
| `domain` | 否 | 固定为 `llm` | 分组/诊断 | 范围过滤，常量不入模 |
| `task` | 是 | 基准任务名称 | 候选事前输入 | 可作粗粒度工作负载类型；不能替代长度等任务特征 |
| `model_id` | 是 | Hugging Face 模型标识 | 标识泄漏风险 | 用于分组、留一模型外验证；不作为默认输入 |
| `nickname` | 是 | 展示名称 | 分组/诊断 | 与 `model_id` 重复，不入模 |
| `architecture` | 是 | Dense、MoE 或混合架构 | 候选事前输入 | 可用，需明确类别编码 |
| `total_params_billions` | 是 | 总参数量（十亿） | 候选事前输入 | 可用；与模型标识相关但具物理解释 |
| `activated_params_billions` | 是 | 每 token 激活参数量（十亿） | 候选事前输入 | MoE 场景重要 |
| `weight_precision` | 是 | 权重精度，如 bfloat16/fp8/mxfp4 | 候选事前输入 | 可用 |
| `gpu_model` | 是 | GPU 型号（本快照仅 H100/B200） | 候选事前输入 | 可用；外推到其他 GPU 不受支持 |
| `num_gpus` | 是 | 本次运行 GPU 数量 | 候选事前输入 | 可用；功率是跨卡聚合值 |
| `max_num_seqs` | 是 | vLLM 最大并发序列数/批容量配置 | 候选事前输入 | 可用；不要误写成实际平均 batch |
| `seed` | 否 | 基准随机种子 | 分组/诊断 | 用于识别重复与复现实验 |
| `num_request_repeats` | 否 | 请求重复次数 | 分组/诊断 | 运行计划已知，但更适合控制/权重，不是任务本征特征 |
| `tensor_parallel` | 是 | 张量并行度 | 候选事前输入 | 可用 |
| `expert_parallel` | 是 | 专家并行度 | 候选事前输入 | 可用，尤其 MoE |
| `data_parallel` | 是 | 数据并行度 | 候选事前输入 | 可用 |
| `steady_state_energy_joules` | 否 | 稳态窗口内全部 GPU 能量求和（J） | 目标 | 只有同时获得预先计划的工作量/持续时间才适合作总能耗目标 |
| `steady_state_duration_seconds` | 否 | 稳态窗口持续时间（s） | 泄漏/运行后结果 | 不作事前输入；用于复核能量=功率×时间 |
| `energy_per_token_joules` | 是 | 稳态 GPU 能量/稳态生成 token 数 | 目标 | 首选强度型目标 |
| `energy_per_request_joules` | 是 | `energy_per_token_joules × avg_output_len` 的估计值 | 目标 | 不是直接逐请求测量；不能用实际输出长度同时作输入 |
| `output_throughput_tokens_per_sec` | 是 | 由稳态能量、每 token 能耗和稳态时长推导的输出吞吐 | 泄漏/运行后结果 | 明确禁用为事前输入 |
| `request_throughput_req_per_sec` | 否 | `output_throughput / avg_output_len` | 泄漏/运行后结果 | 明确禁用为事前输入 |
| `avg_power_watts` | 是 | `steady_state_energy / steady_state_duration`；全部 GPU 聚合 | 目标 | 可作为第二首选目标；比较时必须控制 `num_gpus` |
| `total_output_tokens` | 否 | 全基准实际输出 token 总数 | 泄漏/运行后结果 | 明确禁用为事前输入 |
| `completed_requests` | 否 | 全基准完成请求数 | 泄漏/运行后结果 | 计划请求数与实际完成数必须区分 |
| `avg_output_len` | 是 | `total_output_tokens / completed_requests` | 泄漏/运行后结果 | 高风险泄漏；是模型生成结果，不是预先约束 |
| `mean_itl_ms` | 否 | 平均 inter-token latency | 泄漏/运行后结果 | 仅性能诊断 |
| `median_itl_ms` | 是 | ITL 中位数 | 泄漏/运行后结果 | 仅性能诊断 |
| `p50_itl_ms` | 否 | ITL 第 50 百分位 | 泄漏/运行后结果 | 与 median 语义重复 |
| `p90_itl_ms` | 是 | ITL 第 90 百分位 | 泄漏/运行后结果 | 仅性能诊断 |
| `p95_itl_ms` | 是 | ITL 第 95 百分位 | 泄漏/运行后结果 | 仅性能诊断 |
| `p99_itl_ms` | 是 | ITL 第 99 百分位 | 泄漏/运行后结果 | 仅性能诊断 |
| `avg_batch_size` | 是 | 稳态 Prometheus `vllm:num_requests_running` 均值 | 泄漏/运行后结果 | 实际并发观测，不是运行前配置 |
| `is_stable` | 否 | 是否通过官方稳定性筛选 | 分组/诊断 | 质量过滤，不入模 |
| `unstable_reason` | 否 | 不稳定原因 | 分组/诊断 | 质量分析，不入模 |
| `results_path` | 否 | 对应结果文件相对路径 | 分组/诊断 | 追溯用；可能暴露隐含配置，不入模 |
| `prometheus_path` | 否 | Prometheus 文件相对路径 | 分组/诊断 | 追溯用；不下载时间线 |

## 推荐的最小事前特征集

`task`、`architecture`、`total_params_billions`、`activated_params_billions`、`weight_precision`、`gpu_model`、`num_gpus`、`max_num_seqs`、`tensor_parallel`、`expert_parallel`、`data_parallel`。

`model_id` 只用于分组、重复识别和按模型阻塞/留一模型外验证；若把它直接编码为输入，随机切分很容易得到虚高成绩。

## 明确禁用的输入

所有能耗/功率目标列，以及 `avg_output_len`、`avg_batch_size`、吞吐、ITL、实际完成请求数、实际输出 token 数、稳态持续时间和任何 raw timeline/利用率统计。

## 官方口径依据

- [`LLMRun` 字段定义](https://github.com/ml-energy/data/blob/aaf4d45b8289d96c56c0fb1ebf86dd4d57fac66c/mlenergy/data/records/runs.py#L866-L943)
- [派生指标计算](https://github.com/ml-energy/data/blob/aaf4d45b8289d96c56c0fb1ebf86dd4d57fac66c/mlenergy/data/records/runs.py#L458-L570)
- [稳态边界定义](https://github.com/ml-energy/benchmark/blob/c1d6557fbf7c7c495749b0fcf649f71774eabde9/mlenergy/benchmark/llm/benchmark.py#L209-L263)
- [GPU 能量跨卡求和](https://github.com/ml-energy/benchmark/blob/c1d6557fbf7c7c495749b0fcf649f71774eabde9/mlenergy/benchmark/llm/benchmark.py#L946-L965)
- [leaderboard 公开字段选择](https://github.com/ml-energy/leaderboard/blob/5d899ea5a1c8934c0eff691df2b731d20490bb2a/scripts/build_data.py#L156-L185)
