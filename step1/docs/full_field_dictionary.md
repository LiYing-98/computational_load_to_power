# `runs/llm.parquet` 完整字段字典（文本 LLM inference 子集）

审计粒度：one compiled benchmark run/configuration row。原表 1138 行；严格筛选 `gpqa`、`lm-arena-chat`、`sourcegraph-fim` 后为 694 行、37 列。空值按 Arrow null 统计，空字符串另列。

| 字段 | Arrow 类型 | 分类 | 可进入 `X_pre` | null | 空字符串 | 审计解释 |
|---|---|---|---:|---:|---:|---|
| `activated_params_billions` | `double` | `pre_known` | 是 | 0 | 0 | Activated parameter count is model metadata. |
| `architecture` | `large_string` | `pre_known` | 是 | 0 | 0 | Model architecture is known before launch. |
| `avg_batch_size` | `double` | `post_run_leakage` | 否 | 0 | 0 | Prometheus-observed average concurrent sequences during steady state. |
| `avg_output_len` | `double` | `post_run_leakage` | 否 | 0 | 0 | Actual generated output length is unavailable before execution. |
| `avg_power_watts` | `double` | `target` | 否 | 0 | 0 | Aggregate GPU steady-state power target. |
| `completed_requests` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized completed request count, distinct from planned requests. |
| `data_parallel` | `int64` | `pre_known` | 是 | 0 | 0 | Parallel deployment choice. |
| `domain` | `large_string` | `diagnostic` | 否 | 0 | 0 | Scope filter; constant within this audit. |
| `energy_per_request_joules` | `double` | `target` | 否 | 0 | 0 | Estimated target derived from energy per token and actual output length. |
| `energy_per_token_joules` | `double` | `target` | 否 | 0 | 0 | Measured/derived GPU energy intensity target. |
| `expert_parallel` | `int64` | `pre_known` | 是 | 0 | 0 | Parallel deployment choice. |
| `gpu_model` | `large_string` | `pre_known` | 是 | 0 | 0 | GPU family is selected before launch. |
| `is_stable` | `bool` | `diagnostic` | 否 | 0 | 0 | Post-run quality filter. |
| `max_num_seqs` | `int64` | `pre_known` | 是 | 0 | 0 | Configured maximum concurrent sequences, not observed average batch. |
| `mean_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `median_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `model_id` | `large_string` | `identifier_risk` | 否 | 0 | 0 | Known before launch but can let a model memorize model identity. |
| `nickname` | `large_string` | `diagnostic` | 否 | 0 | 0 | Display label duplicating model identity. |
| `num_gpus` | `int64` | `pre_known` | 是 | 0 | 0 | GPU count is selected before launch. |
| `num_request_repeats` | `int64` | `pre_known` | 是 | 0 | 0 | Planned repetition count determines total request volume before launch. |
| `output_throughput_tokens_per_sec` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized throughput is observed after execution. |
| `p50_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `p90_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `p95_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `p99_itl_ms` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized inter-token latency. |
| `prometheus_path` | `large_string` | `diagnostic` | 否 | 0 | 0 | Provenance path only; timeline is intentionally not downloaded. |
| `request_throughput_req_per_sec` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized request throughput is observed after execution. |
| `results_path` | `large_string` | `diagnostic` | 否 | 0 | 0 | Provenance path for bounded result inspection. |
| `seed` | `int64` | `diagnostic` | 否 | 0 | 0 | Preconfigured reproducibility control; not a physical driver. |
| `steady_state_duration_seconds` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized measurement-window duration is known only after execution. |
| `steady_state_energy_joules` | `double` | `target` | 否 | 0 | 0 | Measured aggregate GPU energy during steady state. |
| `task` | `large_string` | `pre_known` | 是 | 0 | 0 | Benchmark workload is selected before launch. |
| `tensor_parallel` | `int64` | `pre_known` | 是 | 0 | 0 | Parallel deployment choice. |
| `total_output_tokens` | `double` | `post_run_leakage` | 否 | 0 | 0 | Realized full-benchmark output volume. |
| `total_params_billions` | `double` | `pre_known` | 是 | 0 | 0 | Model metadata known before launch. |
| `unstable_reason` | `large_string` | `diagnostic` | 否 | 0 | 565 | Post-run quality explanation. |
| `weight_precision` | `large_string` | `pre_known` | 是 | 0 | 0 | Deployment precision is configured before launch. |

## 键与重复解释

- 37 列整行完全重复：0 组。
- 第一阶段可见的 8 字段键出现 5 个重复组、涉及 10 行。
- 加入 `seed` 与 `num_request_repeats` 后，694 行全部唯一。因此此前的“重复”是工作量重复次数不同，不是无意义重复记录。
- `seed` 在 694 行中恒为 48105，数据不能直接估计跨随机种子的测量波动。

## 口径

四个恒等式校验全部通过：`energy / duration = avg_power`、`energy_per_token × output_throughput = avg_power`、`energy_per_token × avg_output_len = energy_per_request`、`total_output_tokens / completed_requests = avg_output_len`。功率与能量均为稳态窗口内所有 GPU 的聚合 GPU 侧口径，不代表 CPU、整机或数据中心功耗。
